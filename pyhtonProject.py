import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# ── 1. LOAD & CLEAN ──────────────────────────────────────────
df = pd.read_csv('bank.csv', sep=',')
print("Shape:", df.shape)
print(df.isnull().sum())

df['y_binary'] = df['deposit'].map({'yes': 1, 'no': 0})
df['age_group'] = pd.cut(df['age'],
                          bins=[0,25,35,50,100],
                          labels=['18-25','26-35','36-50','50+'])

# ── 2. EDA — CAMPAIGN PERFORMANCE ────────────────────────────
conversion_rate = df['y_binary'].mean() * 100
print(f"Overall campaign conversion rate: {conversion_rate:.2f}%")

# Conversion by job
job_conv = df.groupby('job')['y_binary'].mean().sort_values(ascending=False)

# Conversion by contact month
month_conv = df.groupby('month')['y_binary'].mean()

# Average call duration for converted vs not
dur = df.groupby('deposit')['duration'].mean()
print("Avg call duration:\n", dur)

# ── 3. CUSTOMER SEGMENTATION ─────────────────────────────────
seg = df.groupby('job').agg(
    total_customers=('y_binary', 'count'),
    conversions=('y_binary', 'sum'),
    conversion_rate=('y_binary', 'mean'),
    avg_balance=('balance', 'mean')
).reset_index().sort_values('conversion_rate', ascending=False)

top_segments = seg[seg['conversion_rate'] > seg['conversion_rate'].median()]
revenue_share = top_segments['conversions'].sum() / seg['conversions'].sum() * 100
print(f"Top segments drive {revenue_share:.1f}% of all conversions")

# ── 4. DASHBOARD — 4 CHARTS ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bank Marketing Campaign Performance Dashboard', fontsize=16, fontweight='bold')

# Chart 1 — Conversion by job
axes[0,0].barh(job_conv.index, job_conv.values * 100, color='steelblue')
axes[0,0].set_title('Conversion Rate by Job Type')
axes[0,0].set_xlabel('Conversion Rate (%)')

# Chart 2 — Age group distribution of converted customers
conv_age = df[df['y_binary']==1]['age_group'].value_counts()
axes[0,1].bar(conv_age.index, conv_age.values, color='coral')
axes[0,1].set_title('Converted Customers by Age Group')
axes[0,1].set_ylabel('Count')

# Chart 3 — Call duration vs conversion (box plot)
df.boxplot(column='duration', by='deposit', ax=axes[1,0])
axes[1,0].set_title('Call Duration vs Campaign Outcome')
axes[1,0].set_xlabel('Subscribed (no/yes)')
axes[1,0].set_ylabel('Duration (seconds)')
plt.sca(axes[1,0])
plt.title('Call Duration vs Campaign Outcome')

# Chart 4 — Monthly conversion trend
month_order = ['jan','feb','mar','apr','may','jun',
                'jul','aug','sep','oct','nov','dec']
month_conv_ordered = month_conv.reindex(
    [m for m in month_order if m in month_conv.index])
axes[1,1].plot(month_conv_ordered.index,
               month_conv_ordered.values * 100,
               marker='o', color='green')
axes[1,1].set_title('Monthly Conversion Rate Trend')
axes[1,1].set_ylabel('Conversion Rate (%)')
axes[1,1].set_xlabel('Month')

plt.switch_backend('TkAgg')
plt.tight_layout()
plt.savefig('campaign_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Dashboard saved!")

# ── 5. SQL ANALYSIS ───────────────────────────────────────────
conn = sqlite3.connect(':memory:')
df.to_sql('campaigns', conn, index=False, if_exists='replace')

query1 = """
SELECT job,
       COUNT(*) as total_customers,
       SUM(y_binary) as conversions,
       ROUND(AVG(y_binary)*100, 2) as conversion_rate_pct,
       ROUND(AVG(balance), 2) as avg_balance
FROM campaigns
GROUP BY job
ORDER BY conversion_rate_pct DESC
LIMIT 5
"""
print("\nTop 5 customer segments by conversion rate:")
print(pd.read_sql(query1, conn))

query2 = """
SELECT month,
       COUNT(*) as contacts,
       SUM(y_binary) as conversions,
       ROUND(AVG(y_binary)*100,2) as conversion_rate_pct
FROM campaigns
GROUP BY month
ORDER BY conversion_rate_pct DESC
"""
print("\nCampaign performance by month:")
print(pd.read_sql(query2, conn))

conn.close()

# ── 6. FINAL INSIGHTS ─────────────────────────────────────────
print("\n===== KEY INSIGHTS FOR STAKEHOLDERS =====")
top_job = job_conv.index[0]
top_rate = job_conv.values[0] * 100
print(f"1. Highest converting segment: '{top_job}' at {top_rate:.1f}% conversion rate")
print(f"2. Overall campaign conversion rate: {conversion_rate:.2f}%")
print(f"3. Top customer segments account for {revenue_share:.1f}% of all conversions")
print(f"4. Longer call duration strongly correlates with successful conversion")
print("5. Recommendation: Focus campaign budget on high-converting job segments")
print("   and schedule calls during peak conversion months")