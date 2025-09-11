from io import StringIO
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import glob
from PIL import Image
import requests
import json
import base64

# Page configuration
st.set_page_config(
    page_title="Image Rating System",
    page_icon="📸",
    layout="wide"
)

def upload_to_github(content, filename, commit_message="Update ratings"):
    """Upload file to GitHub repository"""
    try:
        # GitHub configuration from secrets
        github_token = st.secrets["github"]["token"]
        repo_owner = st.secrets["github"]["username"]
        repo_name = st.secrets["github"]["repository"]

        # GitHub API URL
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{filename}"

        # Headers for authentication
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Check if file exists
        response = requests.get(api_url, headers=headers)

        # Encode content to base64
        if isinstance(content, str):
            content_encoded = base64.b64encode(content.encode()).decode()
        else:
            content_encoded = base64.b64encode(content).decode()

        # Prepare data
        data = {
            "message": commit_message,
            "content": content_encoded
        }

        # If file exists, we need the SHA for updating
        if response.status_code == 200:
            existing_file = response.json()
            data["sha"] = existing_file["sha"]
            action = "updated"
        else:
            action = "created"

        # Upload/update file
        response = requests.put(api_url, headers=headers, data=json.dumps(data))

        if response.status_code in [200, 201]:
            file_info = response.json()
            download_url = file_info["content"]["download_url"]
            github_url = file_info["content"]["html_url"]
            return github_url, download_url, action
        else:
            st.error(f"GitHub API Error: {response.status_code} - {response.text}")
            return None, None, None

    except Exception as e:
        st.error(f"Error uploading to GitHub: {e}")
        return None, None, None

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
    """Save a single rating to the CSV file and sync to GitHub"""
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

    # Save to local CSV
    df.to_csv(csv_file, index=False)

    # Auto-sync to GitHub if configured
    if 'github' in st.secrets:
        try:
            csv_content = df.to_csv(index=False)
            github_filename = f"ratings/image_ratings_{datetime.now().strftime('%Y%m%d')}.csv"

            github_url, download_url, action = upload_to_github(
                csv_content,
                github_filename,
                f"Update ratings - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if github_url:
                # Store the URLs in session state for display
                st.session_state.last_github_sync = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'action': action,
                    'github_url': github_url,
                    'download_url': download_url
                }
        except Exception as e:
            st.session_state.github_sync_error = str(e)

    return df

def get_doctor_progress(doctor_name, total_images, df):
    """Get the number of images rated by a doctor"""
    doctor_ratings = df[df['doctor_name'] == doctor_name]
    return len(doctor_ratings['image_name'].unique())

def generate_summary_report(ratings_df):
    """Generate a comprehensive summary report"""
    if ratings_df.empty:
        return "No ratings available yet."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    total_ratings = len(ratings_df)
    unique_images = ratings_df['image_name'].nunique()
    unique_doctors = ratings_df['doctor_name'].nunique()
    avg_rating = ratings_df['rating'].mean()

    # Rating distribution
    rating_dist = ratings_df['rating'].value_counts().sort_index()

    report = f"""
IMAGE RATING SYSTEM - SUMMARY REPORT
Generated: {timestamp}
{'='*50}

OVERVIEW:
- Total Ratings: {total_ratings}
- Unique Images Rated: {unique_images}
- Number of Doctors: {unique_doctors}
- Average Rating: {avg_rating:.2f}/5.0

RATING DISTRIBUTION:
"""

    for rating, count in rating_dist.items():
        percentage = (count / total_ratings * 100)
        report += f"  {rating} stars: {count} ratings ({percentage:.1f}%)\n"

    report += "\nDOCTOR PERFORMANCE:\n" + "-"*30 + "\n"

    for doctor in ratings_df['doctor_name'].unique():
        doctor_data = ratings_df[ratings_df['doctor_name'] == doctor]
        doctor_avg = doctor_data['rating'].mean()
        doctor_count = len(doctor_data)
        doctor_images = doctor_data['image_name'].nunique()

        report += f"""
Doctor: {doctor}
- Images Rated: {doctor_images}
- Total Ratings: {doctor_count}
- Average Rating Given: {doctor_avg:.2f}
"""

    return report

def main():
    st.title("📸 Image Rating System")
    st.markdown("### Rate images on a scale of 1-5")

    # Check GitHub configuration
    github_configured = 'github' in st.secrets

    if github_configured:
        st.success("✅ GitHub integration enabled - ratings will be auto-synced!")

        # Show last sync status
        if 'last_github_sync' in st.session_state:
            sync_info = st.session_state.last_github_sync
            st.info(f"📤 Last sync: {sync_info['timestamp']} - File {sync_info['action']}")
            if sync_info.get('download_url'):
                st.markdown(f"🔗 [Direct Download Link]({sync_info['download_url']})")

        if 'github_sync_error' in st.session_state:
            st.warning(f"⚠️ GitHub sync issue: {st.session_state.github_sync_error}")
    else:
        st.warning("⚠️ GitHub not configured - ratings saved locally only")
        st.info("""
        To enable auto-sync to GitHub (FREE):
        1. Create a GitHub repository
        2. Generate a Personal Access Token
        3. Add GitHub settings to Streamlit secrets
        """)

    # Load images
    image_files = load_images_from_folder()

    if not image_files:
        st.error("No images found! Please place your images in the 'rtdetrv2_pytorch/test_results' folder.")
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

    # Download buttons section
    st.sidebar.subheader("📥 Download Reports")

    # CSV download
    csv = StringIO()
    ratings_df.to_csv(csv, index=False)
    csv.seek(0)
    st.sidebar.download_button(
        label="📊 Download CSV",
        data=csv.getvalue(),
        file_name=f"image_ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # Summary report download
    if not ratings_df.empty:
        summary_report = generate_summary_report(ratings_df)
        st.sidebar.download_button(
            label="📋 Download Summary Report",
            data=summary_report,
            file_name=f"rating_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    # GitHub manual sync and links
    if github_configured and not ratings_df.empty:
        st.sidebar.subheader("🐙 GitHub Backup")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🔄 Sync CSV"):
                with st.spinner("Syncing..."):
                    csv_content = ratings_df.to_csv(index=False)
                    github_filename = f"ratings/image_ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                    github_url, download_url, action = upload_to_github(
                        csv_content,
                        github_filename,
                        f"Manual CSV sync - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    if github_url:
                        st.success("✅ CSV synced!")
                        st.markdown(f"[📁 View on GitHub]({github_url})")
                        st.markdown(f"[⬇️ Direct Download]({download_url})")

        with col2:
            if st.button("📊 Sync Report"):
                with st.spinner("Uploading..."):
                    report_content = generate_summary_report(ratings_df)
                    github_filename = f"reports/rating_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

                    github_url, download_url, action = upload_to_github(
                        report_content,
                        github_filename,
                        f"Summary report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    if github_url:
                        st.success("✅ Report uploaded!")
                        st.markdown(f"[📁 View on GitHub]({github_url})")
                        st.markdown(f"[⬇️ Direct Download]({download_url})")

        # Show persistent download links if available
        if 'last_github_sync' in st.session_state and st.session_state.last_github_sync.get('download_url'):
            st.sidebar.markdown("**🔗 Quick Access:**")
            st.sidebar.markdown(f"[Latest CSV File]({st.session_state.last_github_sync['download_url']})")

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

            st.write(f"**Doctor:** {selected_doctor}")
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

                # Show sync status if GitHub is configured
                if github_configured and 'last_github_sync' in st.session_state:
                    sync_info = st.session_state.last_github_sync
                    st.info(f"🐙 Auto-synced to GitHub at {sync_info['timestamp']}")
                    if sync_info.get('download_url'):
                        st.markdown(f"[📥 Download Latest File]({sync_info['download_url']})")

                st.balloons()

                # Auto-advance to next image
                if st.session_state.current_image_index < total_images - 1:
                    st.session_state.current_image_index += 1
                    st.rerun()
                else:
                    st.success("🎉 All images have been rated!")

        else:
            st.info("Enter your name to start rating images.")

    # Statistics section at the bottom
    if not ratings_df.empty:
        st.subheader("📊 Rating Statistics")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Total Ratings", len(ratings_df))

        with stat_col2:
            st.metric("Images Rated", ratings_df['image_name'].nunique())

        with stat_col3:
            st.metric("Average Rating", f"{ratings_df['rating'].mean():.2f}/5")

        with stat_col4:
            st.metric("Doctors Participating", ratings_df['doctor_name'].nunique())

        # Rating distribution chart
        if len(ratings_df) > 0:
            rating_counts = ratings_df['rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)

if __name__ == "__main__":
    main()