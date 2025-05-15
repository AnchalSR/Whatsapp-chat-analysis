import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import emoji
import numpy as np

def fetch_stats(selected_user, df):
    """Extract basic statistics from the chat data"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Count total messages
    num_messages = df.shape[0]
    
    # Count total words
    words = []
    for message in df['message']:
        words.extend(message.split())
    
    # Count media messages
    num_media_messages = df[df['message'].str.contains('<Media omitted>', case=False, na=False)].shape[0]
    
    # Count links (URLs)
    url_pattern = r'(https?://\S+|www\.\S+)'
    links = []
    for message in df['message']:
        links.extend(re.findall(url_pattern, message))
        
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    """Find most active users in the chat"""
    # Get counts of messages by user
    x = df['user'].value_counts().head()
    
    # Calculate percentages
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'User', 'user': 'Percent of Messages'})
    
    return x, df_percent

def create_wordcloud(selected_user, df):
    """Generate a word cloud from the chat messages"""
    try:
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        # Filter out group notifications and media messages
        temp = df[df['user'] != 'group_notification']
        temp = temp[~temp['message'].str.contains('<Media omitted>', case=False, na=False)]

        # Remove links
        temp['message'] = temp['message'].apply(lambda x: re.sub(r'(https?://\S+|www\.\S+)', '', x))

        def remove_stop_words(message):
            try:
                with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
                    stop_words = f.read().split('\n')
                
                # Remove punctuation and convert to lowercase
                message = re.sub(r'[^\w\s]', '', message.lower())
                
                # Remove stop words
                filtered_words = [word for word in message.split() if word not in stop_words]
                return " ".join(filtered_words)
            except Exception:
                return message

        # Generate word cloud
        wc = WordCloud(width=800, height=500, min_font_size=10, background_color='white', 
                      colormap='viridis', contour_width=1, contour_color='steelblue')
        
        # Apply stop words removal and join all messages
        temp['message'] = temp['message'].apply(remove_stop_words)
        all_words = " ".join(temp['message'].tolist())
        
        # Generate word cloud
        df_wc = wc.generate(all_words if all_words.strip() else "No meaningful words found")
        return df_wc
    except Exception as e:
        print(f"Error creating word cloud: {str(e)}")
        # Return empty wordcloud if there was an error
        return WordCloud().generate("Error")

def most_common_words(selected_user, df):
    """Find most common words in the chat"""
    try:
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        # Filter out group notifications and media messages
        temp = df[df['user'] != 'group_notification']
        temp = temp[~temp['message'].str.contains('<Media omitted>', case=False, na=False)]

        # Remove URLs
        temp['message'] = temp['message'].apply(lambda x: re.sub(r'(https?://\S+|www\.\S+)', '', x))

        # Load stop words
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().split('\n')

        words = []
        for message in temp['message']:
            # Clean the message (remove punctuation and convert to lowercase)
            message = re.sub(r'[^\w\s]', '', message.lower())
            
            for word in message.split():
                if word not in stop_words and len(word) > 1:  # Skip single-character words
                    words.append(word)

        # Create DataFrame of most common words
        most_common_df = pd.DataFrame(Counter(words).most_common(20))
        return most_common_df
    except Exception:
        return pd.DataFrame()  # Return empty DataFrame on error

def emoji_helper(selected_user, df):
    """Analyze emoji usage in the chat"""
    try:
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        emojis = []
        for message in df['message']:
            # Extract all emojis from the message
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

        if not emojis:
            return pd.DataFrame()  # Return empty DataFrame if no emojis found
            
        emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
        return emoji_df
    except Exception:
        return pd.DataFrame()  # Return empty DataFrame on error

def monthly_timeline(selected_user, df):
    """Create monthly timeline of message activity"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Group by year and month, count messages
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    
    # Create a readable time string (e.g., "Jan-2022")
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    
    timeline['time'] = time
    
    # Sort chronologically
    timeline = timeline.sort_values(['year', 'month_num'])
    
    return timeline

def daily_timeline(selected_user, df):
    """Create daily timeline of message activity"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Group by date, count messages
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    
    # Sort chronologically
    daily_timeline = daily_timeline.sort_values('only_date')
    
    return daily_timeline

def week_activity_map(selected_user, df):
    """Create map of activity by day of week"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Order days of week correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['day_name'].value_counts().reindex(day_order)
    
    return counts

def month_activity_map(selected_user, df):
    """Create map of activity by month"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Order months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    counts = df['month'].value_counts().reindex(month_order)
    
    return counts

def activity_heatmap(selected_user, df):
    """Create heatmap of activity by day and time"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create pivot table for heatmap
    user_heatmap = df.pivot_table(
        index='day_name', 
        columns='period', 
        values='message', 
        aggfunc='count'
    ).fillna(0)
    
    # Reindex to ensure days are in correct order
    user_heatmap = user_heatmap.reindex(day_order)
    
    return user_heatmap 