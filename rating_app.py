from io import StringIO

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import glob
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Image Rating System",
    page_icon="📸",
    layout="wide"
)

def load_images_from_folder(folder_path="rtdetrv2_pytorch/test_results"):
    """Load all image files from the specified folder"""
    if not os.path.exists(folder_path):
        return []

    # Support common image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))

    return sorted(image_files)

def load_existing_ratings(csv_file="ratings.csv"):
    """Load existing ratings from CSV file"""
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        # Create empty DataFrame with required columns
        return pd.DataFrame(columns=['image_name', 'doctor_name', 'rating', 'timestamp'])

def save_rating(image_name, doctor_name, rating, csv_file="ratings.csv"):
    """Save a single rating to the CSV file"""
    # Load existing ratings
    df = load_existing_ratings(csv_file)

    # Check if this doctor has already rated this image
    existing_rating = df[(df['image_name'] == image_name) & (df['doctor_name'] == doctor_name)]

    if len(existing_rating) > 0:
        # Update existing rating
        df.loc[(df['image_name'] == image_name) & (df['doctor_name'] == doctor_name), 'rating'] = rating
        df.loc[(df['image_name'] == image_name) & (df['doctor_name'] == doctor_name), 'timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Add new rating
        new_rating = pd.DataFrame({
            'image_name': [image_name],
            'doctor_name': [doctor_name],
            'rating': [rating],
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        df = pd.concat([df, new_rating], ignore_index=True)

    # Save to CSV
    df.to_csv(csv_file, index=False)
    return df

def get_doctor_progress(doctor_name, total_images, df):
    """Get the number of images rated by a doctor"""
    doctor_ratings = df[df['doctor_name'] == doctor_name]
    return len(doctor_ratings['image_name'].unique())

def main():
    st.title("📸 Image Rating System")
    st.markdown("### Rate images on a scale of 1-5")

    # Load images
    image_files = load_images_from_folder()

    if not image_files:
        st.error("No images found! Please place your images in the 'images' folder.")
        st.info("Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF")
        return

    # Load existing ratings
    ratings_df = load_existing_ratings()

    # Sidebar for doctor selection
    st.sidebar.header("Doctor Information")
    selected_doctor = st.sidebar.text_input("Enter your name:")

    # Show progress for selected doctor
    total_images = len(image_files)
    rated_images = get_doctor_progress(selected_doctor, total_images, ratings_df)
    progress = rated_images / total_images

    st.sidebar.metric("Progress", f"{rated_images}/{total_images}")
    st.sidebar.progress(progress)

    ratings_df = load_existing_ratings()
    csv = StringIO()
    ratings_df.to_csv(csv, index=False)
    csv.seek(0)
    st.sidebar.download_button(
        label="Download Result",
        data=csv.getvalue(),
        file_name="results.csv",
        mime="text/csv",
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Images to Rate")

        # Image navigation
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0

        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

        with nav_col1:
            if st.button("← Previous", disabled=(st.session_state.current_image_index == 0)):
                st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)

        with nav_col2:
            st.write(f"Image {st.session_state.current_image_index + 1} of {total_images}")

        with nav_col3:
            if st.button("Next →", disabled=(st.session_state.current_image_index == total_images - 1)):
                st.session_state.current_image_index = min(total_images - 1, st.session_state.current_image_index + 1)

        # Display current image
        if st.session_state.current_image_index < len(image_files):
            current_image_path = image_files[st.session_state.current_image_index]
            image_name = os.path.basename(current_image_path)

            try:
                image = Image.open(current_image_path)
                # Resize image to fit better since it's 480x480
                st.image(image, caption=f"Image: {image_name}", width=480)

            except Exception as e:
                st.error(f"Error loading image: {e}")

    with col2:
        st.subheader("Rate this Image")

        if selected_doctor and st.session_state.current_image_index < len(image_files):
            current_image_path = image_files[st.session_state.current_image_index]
            image_name = os.path.basename(current_image_path)

            # Check if doctor has already rated this image
            existing_rating = ratings_df[
                (ratings_df['image_name'] == image_name) &
                (ratings_df['doctor_name'] == selected_doctor)
                ]

            current_rating = None
            if len(existing_rating) > 0:
                current_rating = int(existing_rating.iloc[0]['rating'])
                st.info(f"Current rating: {current_rating}/5")

            st.write(f"**doctor:** {selected_doctor}")
            st.write(f"**Image:** {image_name}")

            # Rating buttons - 1 to 5
            st.write("**Select your rating:**")

            rating_cols = st.columns(5)
            selected_rating = None

            for i in range(1, 6):
                with rating_cols[i-1]:
                    if st.button(f"{i} ⭐", key=f"rating_{i}", use_container_width=True):
                        selected_rating = i

            # If a rating was selected, save it
            if selected_rating:
                ratings_df = save_rating(image_name, selected_doctor, selected_rating)
                st.success(f"Rating saved: {selected_rating}/5")
                st.balloons()

                # Auto-advance to next image
                if st.session_state.current_image_index < total_images - 1:
                    st.session_state.current_image_index += 1
                    st.rerun()
                else:
                    st.success("🎉 All images have been rated!")

            # Alternative slider method (commented out, but available if preferred)
            # rating = st.select_slider(
            #     "Select rating:",
            #     options=[1, 2, 3, 4, 5],
            #     value=current_rating if current_rating else 3,
            #     format_func=lambda x: f"{x} ⭐" * x
            # )
            # 
            # if st.button("Submit Rating", type="primary"):
            #     ratings_df = save_rating(image_name, selected_doctor, rating)
            #     st.success(f"Rating saved: {rating}/5 for {image_name}")
        else:
            st.info("Enter your name to start rating images.")

if __name__ == "__main__":
    main()