import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import emoji
import helper
import os
import urllib.parse

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide"
)

# Preprocessor function
def preprocess(data):
    # Pattern with flexibility for different date formats
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[APap][Mm])?\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Attempt to parse dates with a more robust approach
    parsed_dates = []
    for date_str in df['message_date']:
        try:
            # Try parsing with AM/PM
            dt = pd.to_datetime(date_str, format='%d/%m/%y, %I:%M %p - ')
        except ValueError:
            try:
                # Try parsing with 24-hour format
                dt = pd.to_datetime(date_str, format='%d/%m/%y, %H:%M - ')
            except ValueError:
                try:
                    # Try with 4-digit year
                     dt = pd.to_datetime(date_str, format='%d/%m/%Y, %I:%M %p - ')
                except ValueError:
                    try:
                        dt = pd.to_datetime(date_str, format='%d/%m/%Y, %H:%M - ')
                    except ValueError:
                        # If all parsing fails, append NaT
                        dt = pd.NaT
                        st.error(f"Could not parse date: {date_str}. Please ensure your chat export is in a standard format.")

        parsed_dates.append(dt)
    
    df['date'] = parsed_dates
    df.dropna(subset=['date'], inplace=True) # Remove rows where date parsing failed

    users = []
    messages_cleaned = []
    for message in df['user_message']:
        entry = re.split('([\\w\\W]+?):\\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages_cleaned.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages_cleaned.append(entry[0])

    df['user'] = users
    df['message'] = messages_cleaned
    df.drop(columns=['user_message', 'message_date'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period

    return df

# Create stop_hinglish.txt if it doesn't exist
def ensure_stop_words_file_exists():
    if not os.path.exists('stop_hinglish.txt'):
        with open('stop_hinglish.txt', 'w') as f:
            stop_words = [
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", 
                "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
                "ka", "ke", "ki", "ko", "main", "me", "mein", "na", "nahi", "ne",
                "par", "se", "the", "they", "this", "to", "tu", "tum", "was", "we",
                "were", "with", "ye", "yeh"
            ]
            f.write("\n".join(stop_words))

# Main app layout
st.sidebar.title("WhatsApp Chat Analyzer")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/512px-WhatsApp.svg.png", width=100)

# Sample data option
st.sidebar.markdown("### Upload your chat")
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat export file (txt)", type="txt")

# Display instructions
with st.sidebar.expander("How to export WhatsApp chat"):
    st.markdown("""
    1. Open WhatsApp and go to your chat
    2. Tap the three dots (⋮) and select 'More'
    3. Select 'Export chat'
    4. Choose 'Without media'
    5. Send the exported file to yourself
    6. Upload that file here
    """)

# Ensure stop words file exists
ensure_stop_words_file_exists()

# Process uploaded file or sample data
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        
        df = preprocess(data)

        # Fetch unique users
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis for", user_list)

        if st.sidebar.button("Show Analysis", key="analyze_btn"):
            with st.spinner("Analyzing the chat data..."):
                # Stats
                num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)
                
                # Display stats in a better format
                st.markdown("### 📊 Chat Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Messages", num_messages)
                with col2:
                    st.metric("Total Words", num_words)
                with col3:
                    st.metric("Media Shared", num_media)
                with col4:
                    st.metric("Links Shared", num_links)
                
                # Monthly timeline
                st.markdown("### 📅 Monthly Activity")
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(timeline['time'], timeline['message'], color='#25D366', marker='o')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # Daily timeline
                st.markdown("### 📆 Daily Activity")
                daily_timeline_df = helper.daily_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(daily_timeline_df['only_date'], daily_timeline_df['message'], color='#128C7E', marker='.')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Activity maps
                st.markdown("### 🗓️ Activity Patterns")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Most Active Days")
                    busy_day = helper.week_activity_map(selected_user, df)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.bar(busy_day.index, busy_day.values, color='#075E54')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.markdown("#### Most Active Months")
                    busy_month = helper.month_activity_map(selected_user, df)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.bar(busy_month.index, busy_month.values, color='#128C7E')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Heatmap
                st.markdown("### 🔥 Weekly Activity Heatmap")
                user_heatmap = helper.activity_heatmap(selected_user, df)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax = sns.heatmap(user_heatmap, cmap="YlGnBu")
                st.pyplot(fig)

                # Most busy users (group level)
                if selected_user == 'Overall':
                    st.markdown("### 👥 Most Active Members")
                    x, new_df = helper.most_busy_users(df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.bar(x.index, x.values, color='#34B7F1')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    with col2:
                        st.dataframe(new_df, use_container_width=True)

                # Word Cloud
                st.markdown("### ☁️ Word Cloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(df_wc)
                ax.axis("off")
                st.pyplot(fig)

                # Most common words
                st.markdown("### 🔠 Most Common Words")
                most_common_df = helper.most_common_words(selected_user, df)
                if not most_common_df.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.barh(most_common_df[0], most_common_df[1], color='#075E54')
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No common words found after filtering stop words.")

                # Emoji analysis
                st.markdown("### 😀 Emoji Analysis")
                emoji_df = helper.emoji_helper(selected_user, df)
                
                if not emoji_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(emoji_df.rename(columns={0: "Emoji", 1: "Count"}), use_container_width=True)
                    with col2:
                        # Limit to top 10 emojis for the pie chart
                        top_emojis = emoji_df.head(10)
                        fig, ax = plt.subplots(figsize=(7, 7))
                        ax.pie(top_emojis[1], labels=top_emojis[0], autopct="%0.2f%%", shadow=True)
                        st.pyplot(fig)
                else:
                    st.info("No emojis found in the selected messages.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you've uploaded a valid WhatsApp chat export file.")
else:
    # Display a welcome message when no file is uploaded
    st.markdown("""
    ## 👋 Welcome to WhatsApp Chat Analyzer!
    
    This app helps you analyze your WhatsApp conversations and discover interesting insights.
    
    **To get started:**
    1. Export a chat from WhatsApp (without media)
    2. Upload the .txt file using the sidebar
    3. Select a user to analyze (or "Overall" for the entire chat)
    4. Click "Show Analysis"
    
    **You'll discover:**
    - Message statistics and trends
    - Activity patterns by time of day, day of week, and month
    - Most common words and emojis
    - Beautiful visualizations of your chat data
    
    Upload a chat file from the sidebar to begin!
    """)

# Footer
st.markdown("---")
st.markdown("WhatsApp Chat Analyzer © 2023 | Made with Streamlit")










