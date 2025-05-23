import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

# === Load Data ===
@st.cache_data
def load_data():
    first_df = pd.read_csv("clean_read_first_time.csv", parse_dates=['datetime'])
    returning_df = pd.read_csv("clean_read_returning.csv", parse_dates=['datetime'])
    subscribe_df = pd.read_csv("clean_subscribe.csv", parse_dates=['datetime'])
    buy_df = pd.read_csv("clean_buy.csv", parse_dates=['datetime'])
    return first_df, returning_df, subscribe_df, buy_df

first_df, returning_df, subscribe_df, buy_df = load_data()
read_df = pd.concat([first_df, returning_df], ignore_index=True)

# === Sidebar Filters ===
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", read_df['datetime'].min().date())
end_date = st.sidebar.date_input("End Date", read_df['datetime'].max().date())
source_options = first_df['source'].unique().tolist()
selected_sources = st.sidebar.multiselect("Source", options=source_options, default=source_options)
selected_countries = st.sidebar.multiselect("Country", options=read_df['country'].unique(), default=list(read_df['country'].unique()))
selected_topics = st.sidebar.multiselect("Topic", options=read_df['topic'].unique(), default=list(read_df['topic'].unique()))

# === Filter Function ===
def apply_filters(df):
    return df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]

first_df_f = apply_filters(first_df)
first_df_f = first_df_f[first_df_f['source'].isin(selected_sources) &
                        first_df_f['country'].isin(selected_countries) &
                        first_df_f['topic'].isin(selected_topics)]

returning_df_f = apply_filters(returning_df)
returning_df_f = returning_df_f[returning_df_f['country'].isin(selected_countries) &
                                returning_df_f['topic'].isin(selected_topics)]

read_df_f = pd.concat([first_df_f, returning_df_f], ignore_index=True)
subscribe_df_f = apply_filters(subscribe_df)
buy_df_f = apply_filters(buy_df)

# === KPIs ===
st.title("Dilan's Travel Blog Funnel Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Reads", len(read_df_f))
col2.metric("Unique Readers", read_df_f['user_id'].nunique())
col3.metric("Returning %", f"{(len(returning_df_f) / len(read_df_f) * 100):.1f}%")
read_to_sub = (subscribe_df_f['user_id'].nunique() / read_df_f['user_id'].nunique()) if read_df_f['user_id'].nunique() else 0

col4, col5, col6 = st.columns(3)
col4.metric("Total Subscriptions", len(subscribe_df_f))
conv = (buy_df_f['user_id'].nunique() / subscribe_df_f['user_id'].nunique()) if subscribe_df_f['user_id'].nunique() else 0
col5.metric("Sub→Buy Conversion", f"{conv * 100:.1f}%")
col6.metric("Total Revenue", f"${buy_df_f['price'].sum():.2f}")
col7, _ = st.columns([1, 2])
col7.metric("Reader→Sub Conversion", f"{read_to_sub * 100:.1f}%")

# === Reads by Source ===
st.subheader("Reads by Source")
df_source = first_df_f.groupby('source').size().reset_index(name='count')
df_source = df_source.sort_values(by='count', ascending=False)
fig1 = px.bar(
    df_source,
    x='source',
    y='count',
    text='count',
    title='Reads by Source (First-Time Users)'
)
fig1.update_traces(
    marker_color='steelblue',
    textposition='outside',
    textfont=dict(size=22, color='dimgray')  # Bigger, readable numbers
)

fig1.update_layout(
    xaxis=dict(title="Source", showgrid=False, showline=False, zeroline=False),
    yaxis=dict(title="Read Count", showgrid=False, showline=False, zeroline=False),
    title_font_size=22,
    title_x=0.5,
    plot_bgcolor='white',
    margin=dict(t=60, b=40, l=40, r=40),
    font=dict(size=16)  # Controls axis and tick label size
)
st.markdown("""
### Reddit generated the highest volume of first-time readers.  
These are strong acquisition channels worth reinforcing with more budget or tailored content.
""")
st.plotly_chart(fig1, use_container_width=True)

# === Reads by Topic ===
st.subheader("Reads by Topic")
topic_df = read_df_f.groupby('topic').size().reset_index(name='count')
topic_df = topic_df.sort_values(by='count', ascending=False)
fig2 = px.bar(topic_df, x='topic', y='count', text='count', title='Reads by Topic')
fig2.update_layout(
    xaxis_title="Topic",
    yaxis_title="Read Count",
    title_font_size=20,
    title_x=0.5,
    plot_bgcolor='white'
)
fig2.update_traces(marker_color='darkorange', textposition='outside', textfont_size=20)
fig2.update_layout(
    xaxis=dict(showgrid=False, showline=False, zeroline=False),
    yaxis=dict(showgrid=False, showline=False, zeroline=False),
    plot_bgcolor='white'
)
st.plotly_chart(fig2)
top_topic_row = topic_df.iloc[0]
st.markdown(f"The topic **{top_topic_row['topic']}** received the most interest with **{top_topic_row['count']:,}** reads. Expanding content on this topic could boost traffic and engagement.")

# === Reader Overlap as Pie Chart ===
st.subheader("Reader Overlap Across Topics")
user_topics = read_df_f.groupby('user_id')['topic'].nunique().reset_index(name='topic_count')
topic_counts = user_topics['topic_count'].value_counts(normalize=True).sort_index().reset_index()
topic_counts.columns = ['Topics Read', 'Percentage']
topic_counts['Percentage'] = topic_counts['Percentage'] * 100
fig_pie = px.pie(
    topic_counts,
    names='Topics Read',
    values='Percentage',
    title="Distribution of Users by Number of Topics Read",
    hole=0.4
)
fig_pie.update_layout(
    title_font_size=20,
    title_x=0.5
)
fig_pie.update_traces(textinfo='percent+label', textfont_size=15)
st.plotly_chart(fig_pie)
st.markdown("A significant portion of users engage with only one topic — almost two-thirds of them.")

# === Revenue by Country ===
st.subheader("Revenue by Country")
first_country_map = read_df.sort_values('datetime').drop_duplicates('user_id')[['user_id', 'country']]
buy_users = buy_df_f.merge(first_country_map, on='user_id', how='left')
rev_country = buy_users.groupby('country')['price'].sum().reset_index()
rev_country = rev_country.sort_values(by='price', ascending=False)
fig3 = px.bar(rev_country, x='country', y='price', text='price', title='Revenue by Country')
fig3.update_layout(
    xaxis_title="Country",
    yaxis_title="Total Revenue ($)",
    title_font_size=20,
    title_x=0.5,
    plot_bgcolor='white'
)
fig3.update_traces(marker_color='seagreen', textposition='outside', textfont_size=20)
fig3.update_layout(
    xaxis=dict(showgrid=False, showline=False, zeroline=False),
    yaxis=dict(showgrid=False, showline=False, zeroline=False),
    plot_bgcolor='white'
)
st.plotly_chart(fig3)
top_rev_country = rev_country.iloc[0]
st.markdown(f"Users from **{top_rev_country['country']}** generated the highest revenue (${top_rev_country['price']:.2f}). It may be strategic to localize offers or create region-specific campaigns.")

# === Product Segmentation ===
st.subheader("Product Segmentation")
buy_df_f['product'] = buy_df_f['price'].apply(lambda x: "E-book" if x == 8 else "Course")
product_seg = buy_df_f.groupby('product').agg(
    user_count=('user_id', 'nunique'),
    revenue=('price', 'sum')
).reset_index()
product_seg = product_seg.sort_values(by='revenue', ascending=False)
fig_prod = px.bar(product_seg, x='product', y='revenue', text='user_count',
                  title="Revenue & Buyers by Product")
fig_prod.update_layout(
    xaxis_title="Product",
    yaxis_title="Revenue ($)",
    title_font_size=20,
    title_x=0.5,
    plot_bgcolor='white'
)
fig_prod.update_traces(marker_color='purple', textposition='outside', textfont_size=20)
fig_prod.update_layout(
    xaxis=dict(showgrid=False, showline=False, zeroline=False),
    yaxis=dict(showgrid=False, showline=False, zeroline=False),
    plot_bgcolor='white'
)
st.plotly_chart(fig_prod)
st.markdown("The \$80 **Course** and \$8 **E-book** generate revenue differently. Consider pushing whichever performs better across countries or topics.")

# === Product Revenue by Country ===
st.subheader("Product Revenue by Country")
prod_country = buy_df_f.merge(first_country_map, on='user_id', how='left')
prod_country_grouped = prod_country.groupby(['product', 'country'])['price'].sum().reset_index()

country_order = prod_country_grouped.groupby('country')['price'].sum().sort_values(ascending=False).index.tolist()
prod_country_grouped['country'] = pd.Categorical(prod_country_grouped['country'], categories=country_order, ordered=True)
prod_country_grouped = prod_country_grouped.sort_values(by='country')
fig_country_prod = px.bar(prod_country_grouped, x='country', y='price', color='product',
                          title="Product Revenue by Country", barmode='group')
fig_country_prod.update_layout(
    xaxis_title="Country",
    yaxis_title="Revenue ($)",
    title_font_size=20,
    title_x=0.5,
    legend_title_text='Product',
    plot_bgcolor='white'
)
fig_country_prod.update_layout(
    xaxis=dict(showgrid=False, showline=False, zeroline=False),
    yaxis=dict(showgrid=False, showline=False, zeroline=False),
    plot_bgcolor='white'
)
st.plotly_chart(fig_country_prod)

# === Product Revenue by Topic ===
st.subheader("Product Revenue by Topic")
first_topic_map = read_df.sort_values('datetime').drop_duplicates('user_id')[['user_id', 'topic']]
prod_topic = buy_df_f.merge(first_topic_map, on='user_id', how='left')
prod_topic_grouped = prod_topic.groupby(['product', 'topic'])['price'].sum().reset_index()

topic_order = prod_topic_grouped.groupby('topic')['price'].sum().sort_values(ascending=False).index.tolist()
prod_topic_grouped['topic'] = pd.Categorical(prod_topic_grouped['topic'], categories=topic_order, ordered=True)
prod_topic_grouped = prod_topic_grouped.sort_values(by='topic')
fig_topic_prod = px.bar(prod_topic_grouped, x='topic', y='price', color='product',
                        title="Product Revenue by Topic", barmode='group')
fig_topic_prod.update_layout(
    xaxis_title="Topic",
    yaxis_title="Revenue ($)",
    title_font_size=20,
    title_x=0.5,
    legend_title_text='Product',
    plot_bgcolor='white'
)
fig_topic_prod.update_layout(
    xaxis=dict(showgrid=False, showline=False, zeroline=False),
    yaxis=dict(showgrid=False, showline=False, zeroline=False),
    plot_bgcolor='white'
)
st.plotly_chart(fig_topic_prod)

# === Conversion Funnel ===
st.subheader("Conversion Funnel")
funnel_data = pd.DataFrame({
    'Stage': ['Readers', 'Subscribers', 'Buyers'],
    'Users': [
        read_df_f['user_id'].nunique(),
        subscribe_df_f['user_id'].nunique(),
        buy_df_f['user_id'].nunique()
    ]
})
funnel_data = funnel_data.sort_values(by='Users', ascending=False)
fig4 = px.funnel(funnel_data, x='Users', y='Stage', title='User Conversion Funnel')
fig4.update_traces(
    textfont_size=20  # Increase value label size
)

fig4.update_layout(
    xaxis_title="Users",
    yaxis_title="Funnel Stage",
    title_font_size=20,
    title_x=0.5,
    plot_bgcolor='white'
)
st.plotly_chart(fig4)
st.markdown("The largest drop in the funnel is from **readers to subscribers**. Improving sign-up CTAs or value propositions could help increase the subscription rate.")

# === Rolling Retention & Churn ===
st.subheader("Rolling Retention & Churn Rates")
read_sorted = read_df.sort_values(by=['user_id', 'datetime'])
first_reads = read_sorted.groupby('user_id')['datetime'].min().reset_index().rename(columns={'datetime': 'first_read'})
merged = read_sorted.merge(first_reads, on='user_id')
merged['days_since_first'] = (merged['datetime'] - merged['first_read']).dt.days
retention_curve = merged.groupby('days_since_first')['user_id'].nunique().reset_index()
retention_curve['retention_rate'] = retention_curve['user_id'] / merged['user_id'].nunique()

ret_day = {d: retention_curve.query(f'days_since_first == {d}')['retention_rate'].values[0] if d in retention_curve['days_since_first'].values else 0 for d in [1, 7, 14, 30]}
churn_day = {d: 1 - ret_day[d] for d in ret_day}

ret_cols = st.columns(4)
for idx, d in enumerate([1, 7, 14, 30]):
    ret_cols[idx].metric(f"Day {d} Retention", f"{ret_day[d]*100:.2f}%")

churn_cols = st.columns(4)
for idx, d in enumerate([1, 7, 14, 30]):
    delta_val = (churn_day[d] - churn_day[1]) * 100 if d != 1 else None
    churn_cols[idx].metric(f"Day {d} Churn", f"{churn_day[d]*100:.2f}%",
                           delta=f"{delta_val:.2f}%" if delta_val is not None else None,
                           delta_color="inverse")
st.markdown("Retention falls sharply after Day 1. A drip email campaign or retargeting ads could help re-engage users during these drop-off points.")


# === Retention & Churn Line Chart ===
st.subheader("User Retention and Churn Trends")

# Data setup
days = [1, 7, 14, 30]
ret_vals = [ret_day[d] * 100 for d in days]
churn_vals = [churn_day[d] * 100 for d in days]

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=days,
    y=ret_vals,
    mode='lines+markers+text',
    name='Retention (%)',
    line=dict(color='seagreen', width=3),
    text=[f"{v:.1f}%" for v in ret_vals],
    textposition="top center",
    textfont=dict(size=18, color='seagreen')
))

fig_line.add_trace(go.Scatter(
    x=days,
    y=churn_vals,
    mode='lines+markers+text',
    name='Churn (%)',
    line=dict(color='indianred', width=3, dash='dash'),
    text=[f"{v:.1f}%" for v in churn_vals],
    textposition="bottom center",
    textfont=dict(size=18, color='indianred')
))

fig_line.update_layout(
    title='User Retention and Churn Over Time',
    xaxis_title='Day Since First Visit',
    yaxis_title='Percentage (%)',
    xaxis=dict(tickmode='array', tickvals=days, showgrid=False, showline=False, zeroline=False),
    yaxis=dict(showgrid=False, showline=False, zeroline=False),
    plot_bgcolor='white',
    title_x=0.5,
    font=dict(size=16),
    margin=dict(t=60, b=40, l=40, r=40),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig_line, use_container_width=True)

# Explanation
st.markdown("""
**What do these metrics mean?**

- **Retention Rate**: The percentage of users who returned to the blog on a specific day after their first visit.  
  For example, a Day 1 retention rate of 10% means that 10% of users came back the next day.

- **Churn Rate**: The percentage of users who did **not** return by a specific day.  
  For instance, a Day 30 churn rate of 99.35% means nearly all users dropped off after their first visit.

**Suggestion:** The steep drop-off after Day 1 suggests a need for re-engagement strategies like drip emails, push notifications, or targeted retargeting ads.
""")

# === ROI Table ===
st.subheader("Marketing Channel ROI (3-Month Estimate)")
spend_map = {'AdWords': 1500, 'SEO': 750, 'Reddit': 750}
reader_counts = first_df.groupby('source')['user_id'].nunique()
buyer_counts = buy_df.merge(first_df[['user_id', 'source']], on='user_id', how='left')
buyers_by_source = buyer_counts.groupby('source')['user_id'].nunique()
revenue_by_source = buyer_counts.groupby('source')['price'].sum()

roi_df = pd.DataFrame({
    'Source': spend_map.keys(),
    'Spend': [spend_map[s] for s in spend_map],
    'Readers': [reader_counts.get(s, 0) for s in spend_map],
    'Buyers': [buyers_by_source.get(s, 0) for s in spend_map],
    'Revenue': [revenue_by_source.get(s, 0) for s in spend_map]
})

roi_df['CR (%)'] = (roi_df['Buyers'] / roi_df['Readers'] * 100).round(2)
roi_df['CPC ($)'] = (roi_df['Spend'] / roi_df['Readers']).round(2)
roi_df['CPA ($)'] = (roi_df['Spend'] / roi_df['Buyers']).round(2)
roi_df['ROI (%)'] = (((roi_df['Revenue'] - roi_df['Spend']) / roi_df['Spend']) * 100).round(2)

st.dataframe(roi_df)
top_roi = roi_df.sort_values(by="ROI (%)", ascending=False).iloc[0]
top_conv = roi_df.sort_values(by="CR (%)", ascending=False).iloc[0]
st.markdown(f"`{top_roi['Source']}` delivered the highest ROI (**{top_roi['ROI (%)']:.2f}%**). Allocate more budget here to maximize returns. "
            f"`{top_conv['Source']}` showed the best reader-to-buyer conversion rate (**{top_conv['CR (%)']:.2f}%**) — ideal for driving high-intent users.")

# === ROI chart ===
st.subheader("Why Reddit Should Lead the Marketing Mix")

# Filter absolute metrics
abs_metrics = roi_df[['Source', 'Revenue', 'Buyers']].melt(id_vars='Source', var_name='Metric', value_name='Value')
abs_metrics = abs_metrics.sort_values(by='Value', ascending=False)

fig_abs = px.bar(
    abs_metrics,
    x='Source',
    y='Value',
    color='Metric',
    barmode='group',
    title='Revenue & Buyers by Channel'
)

fig_abs.update_traces(texttemplate='%{value:.0f}', textposition='outside', textfont_size=16)
fig_abs.update_layout(
    xaxis=dict(title="Channel", showgrid=False, showline=False, zeroline=False),
    yaxis=dict(title="Value", showgrid=False, showline=False, zeroline=False),
    title_font_size=20,
    title_x=0.5,
    legend_title_text='Metric',
    plot_bgcolor='white',
    font=dict(size=14),
    margin=dict(t=60, b=40, l=40, r=40)
)
st.plotly_chart(fig_abs)

# Filter % metrics
pct_metrics = roi_df[['Source', 'ROI (%)', 'CR (%)']].melt(id_vars='Source', var_name='Metric', value_name='Value')
pct_metrics = pct_metrics.sort_values(by='Value', ascending=False)

fig_pct = px.bar(
    pct_metrics,
    x='Source',
    y='Value',
    color='Metric',
    barmode='group',
    title='ROI and Conversion Rates by Channel'
)

fig_pct.update_traces(texttemplate='%{value:.1f}%', textposition='outside', textfont_size=16)
fig_pct.update_layout(
    xaxis=dict(title="Channel", showgrid=False, showline=False, zeroline=False),
    yaxis=dict(title="Percentage (%)", showgrid=False, showline=False, zeroline=False),
    title_font_size=20,
    title_x=0.5,
    legend_title_text='Metric',
    plot_bgcolor='white',
    font=dict(size=14),
    margin=dict(t=60, b=40, l=40, r=40)
)
st.plotly_chart(fig_pct)

st.markdown("""
Reddit stands out as the top performer across both absolute and percentage-based metrics:
- **Highest revenue and number of buyers**
- **Best ROI (%)** overall
This makes it the strongest candidate for primary focus in the marketing mix.
""")