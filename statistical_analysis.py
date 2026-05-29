import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.sankey as sankey
from wordcloud import WordCloud
from transformers import pipeline
from transformers import logging
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoProcessor, Gemma3nForConditionalGeneration
from datasets import Dataset as Dataset_hf
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.special import softmax
import os 
from tqdm import tqdm
from typing import List
from scipy.stats import ttest_ind, mannwhitneyu, levene, normaltest
import seaborn as sns
import random
import argparse
import logging
from collections import Counter

# Logger for report
rep_logger = logging.getLogger("report_logger")
rep_logger.addHandler(logging.FileHandler("report.log", mode="w"))

# Logger for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='analysis.log', filemode='w')

matplotlib.rc({'font.size': 14})
plt.rcParams.update({'font.size': 14})

def load_and_merge_data(platform="bluesky"):
    ''' Function to load and merge default labelled data and raw data, and save merged data as csv for future use to avoid having to merge every time. If merged csv already exists, load that instead.'''

    if platform == "bluesky":
        if not os.path.exists("bluesky_full_info_RecollectionApr28.csv"):
            labelled_df = pd.read_csv("exp10_mixed_weak_gold_deberta-v3-small_predictions_bluesky_posts_RecollectionApr28_cleaned.csv")
            raw_df = pd.read_csv("bluesky_posts_RecollectionApr28_cleaned.csv")
            original_df = raw_df.merge(labelled_df[["comment_id", "pred_contains_appearance", "pred_sub_category"]], left_on="item_id", right_on="comment_id", how="left")

            print(len(original_df))

            original_df["created_at"] = original_df["created_at"].apply(lambda x: pd.to_datetime(x, errors='coerce')) # Force created_at to be datetime, coercing errors to NaT

            original_df.to_csv("bluesky_full_info_RecollectionApr28.csv", index=False)

        else:
            original_df = pd.read_csv("bluesky_full_info_RecollectionApr28.csv")
    elif platform == "twitter":
        pass # To Do: Implement for X/Twitter
    else:
        raise ValueError("Platform not supported. Please choose 'bluesky' or 'twitter'.")
    
    return original_df

def dataset_summary(df):
    ''' Function to compute summary statistics for the dataset. '''
    print("Dataset summary:")
    print(f"Total number of observations: {len(df)}")
    print(f"Number of unique MPs: {df['mp_handle'].nunique()}")
    print(f"Number of MPs by gender: {df.groupby('gender')['mp_handle'].nunique()}")
    print(f"Number of unique authors: {df['author'].nunique()}")

def load_mp_details_and_merge(original_df):
    print("======== MP DETAILS ========")
    mp_details = pd.read_csv("mp_details_full_info.csv", dtype={"theyworkforyou_id": str, "parliament_id": str, "wiki_birth_year": str})
    mp_details = mp_details.replace("—", np.nan)
    original_df = original_df.merge(mp_details[["bluesky_handle", "minority_status", "ethnicity", "party", "wiki_birth_year"]], left_on="mp_handle", right_on="bluesky_handle", how="left")
    return original_df, mp_details

def mp_post_analysis(original_df):
    print("======== MP POST ANALYSIS ========")
    print("MPs grouped by number of posts:")
    posts_only = original_df[original_df["item_type"] == "post"]
    print(f"Number of MPs with posts: {posts_only['mp_handle'].nunique()}")
    post_counts = posts_only.groupby("mp_handle").size().reset_index(name="post_count")
    print(post_counts["post_count"].median())
    print("Number of posts by duplicate status:")
    print(posts_only.groupby("duplicates")["item_id"].count())
    return post_counts

def proportion_replies_by_gender(original_df):
    print("======== PROPORTION OF REPLIES BY GENDER ========")
    print(f"Length of dataset: {len(original_df)}")
    print("Proportion of replies by gender:")
    replies_to_women = original_df[original_df["gender"] == "Female"]
    print(len(replies_to_women)/len(original_df) * 100)
    

def duplicates_analysis(original_df, label="pred_contains_appearance"):
    print("======== DUPLICATES ANALYSIS ========")
    print(f"Proportion of {label} in replies by duplicates:")
    print(original_df.groupby("duplicates")[label].mean()*100)
    print(f"Proportion of {label} comments in replies by duplicates:")
    print(original_df.groupby("duplicates")[label].mean()*100)

def mp_reply_analysis(original_df):
    ''' Function to analyze replies to MPs '''
    print("======== MP REPLY ANALYSIS ========")
    assert all(original_df["item_type"] == "reply"), "Not all items in original_df are replies."
    print(f"Number of MPs with replies: {original_df['mp_handle'].nunique()}")
    reply_counts = original_df.groupby("mp_handle").size().reset_index(name="reply_count")
    reply_counts["pred_contains_appearance"] = original_df.groupby("mp_handle")["pred_contains_appearance"].sum().values
    return reply_counts

def plot_reply_distribution(reply_counts):
    ''' Function to plot the distribution of replies by MPs '''
    print(f"==== DISTRIBUTION OF REPLIES ========")
    reply_counts = reply_counts[reply_counts["reply_count"] <= 10000] # Filter out MPs with more than 1000 replies for better visualization
    plt.figure(figsize=(10, 6))
    plt.hist(reply_counts["reply_count"], bins=200, color='blue', alpha=0.7)
    plt.title("Distribution of Replies by MPs")
    plt.xlabel("Number of Replies")
    plt.ylabel("Number of MPs")
    plt.axvline(reply_counts["reply_count"].median(), color='red', linestyle='dashed', linewidth=1)
    plt.show()
    plt.savefig("replies_distribution.png")


def compare_demographics_by_engagement_level(reply_counts):
    print("======== COMPARING DEMOGRAPHICS BY ENGAGEMENT LEVEL ========")
    print("Demographics by engagement level:")
    # Chi square test comparing proportion of women in active vs inactive MPs
    contingency_table = pd.crosstab(reply_counts["active_filter"], reply_counts["gender"])
    print("Contingency table for active vs inactive MPs by gender:")
    print(contingency_table)
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("Chi-square test results comparing proportion of women in active vs inactive MPs:")
    print(f"Chi2: {chi2}, p-value: {p}, degrees of freedom: {dof}")
    print("Expected frequencies:")
    print(expected)

    # Chi square test comparing proportion of ethnic minority MPs in active vs inactive MPs
    contingency_table = pd.crosstab(reply_counts["active_filter"], reply_counts["minority_status"])
    print("Contingency table for active vs inactive MPs by minority status:")
    print(contingency_table)
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("Chi-square test results comparing proportion of ethnic minority MPs in active vs inactive MPs:")
    print(f"Chi2: {chi2}, p-value: {p}, degrees of freedom: {dof}")
    print("Expected frequencies:")
    print(expected)

    # Mann-Whitney U test comparing age of active vs inactive MPs
    print(f"Age distribution by active vs inactive MPs:")
    reply_counts["wiki_birth_year"] = pd.to_numeric(reply_counts["wiki_birth_year"], errors="coerce")
    print(reply_counts.groupby("active_filter")["wiki_birth_year"].describe())
    active_ages = reply_counts[reply_counts["active_filter"] == True]["wiki_birth_year"].dropna()
    inactive_ages = reply_counts[reply_counts["active_filter"] == False]["wiki_birth_year"].dropna()
    results = mannwhitneyu(active_ages, inactive_ages, alternative="two-sided")
    print("Mann-Whitney U test results comparing age of active vs inactive MPs:")
    print(results)  

def mannwhitney_by_group(reply_counts, label="reply_count", group_by="gender"):
    ''' Function to run a Mann-Whitney U test comparing the distribution of a specified label (e.g. reply_count) between two groups defined by a specified demographic variable (e.g. gender, minority_status) '''
    print(f"======== COMPARING {label.upper()} BY {group_by.upper()} ========")
    assert all(reply_counts["active_filter"] == True), "Not all MPs in df are active. Please filter for active MPs before comparing"
    mean_replies_by_gender = reply_counts.groupby(group_by)[label].mean()
    print(f"Mean {label} by {group_by}:")
    print(mean_replies_by_gender)
    #print(f"Mean replies by gender excluding Lizzi Collinge: {active_reply_counts[active_reply_counts['mp_handle'] != 'lizzicollinge.bsky.social'].groupby("gender")['reply_count'].mean()}")
    if group_by == "gender":
        group_a = "Male"
        group_b = "Female"
    elif group_by == "minority_status":
        group_a = "Minority"
        group_b = "Unknown"
    else:
        raise ValueError("Group by variable not supported. Please choose 'gender' or 'minority_status'.")
    a_replies = reply_counts[reply_counts[group_by] == group_a][label].dropna()
    b_replies = reply_counts[reply_counts[group_by] == group_b][label].dropna()
    results = mannwhitneyu(a_replies, b_replies, alternative="two-sided")
    print(f"Mann-Whitney U test results comparing {label} by {group_by}:")
    print(results)


def appearance_by_gender(original_df, reply_counts):
    print("========= APPEARANCE RELATED REPLIES =========")
    cross_tab = pd.crosstab(original_df["pred_contains_appearance"], original_df["gender"], normalize="columns")
    print(cross_tab)
    print("Mean percentage of appearance related replies by gender:")
    print(reply_counts.groupby("gender")["proportion_appearance_replies"].mean())
    print("Mean number of appearance related replies by gender:")
    print(reply_counts.groupby("gender")["pred_contains_appearance"].mean())

def plot_appearance_subtypes_by_gender(original_df):
    ''' Function to plot subtypes of appearance related replies by gender '''
    cross_tab = pd.crosstab(original_df["pred_sub_category"], original_df["gender"], normalize="columns")
    cross_tab = cross_tab.transpose()
    cross_tab.plot(kind="bar", color={"Male": "blue", "Female": "lightblue"}, figsize=(10, 8))
    plt.title("Subtypes of Appearance Related Replies by Gender")
    plt.xlabel("Subtype of Appearance Related Reply")
    plt.ylabel("Proportion of Appearance Related Replies")
    plt.legend(title="Gender")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("appearance_subtypes_by_gender.png")

def appearance_subtypes_by_gender(original_df):
    ''' Function to analyse subtypes of appearance related replies by gender '''
    print("Subtypes of appearance related replies by gender:")

    appearance_subcategories = original_df["pred_sub_category"].dropna().unique()

    appearance_df = original_df[original_df["pred_contains_appearance"] == 1]

    cross_tab = pd.crosstab(appearance_df["gender"], appearance_df["pred_sub_category"], normalize="index")
    print(cross_tab) # CHI square not appropriate as data points not independent 

    plot_appearance_subtypes_by_gender(original_df)

def plot_reply_by_appearance(active_reply_counts):
    ''' Function to plot percentage of appearance related replies by number of replies, colored by gender '''
    # Scatter plot of percentage of appearance related replies by number of replies, colored by gender
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="reply_to_post_ratio", y="proportion_appearance_replies", data=active_reply_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    # x scale logarithmic
    plt.xscale("log")
    plt.title("Percentage of Appearance Related Replies by Replies to Post Ratio")
    plt.xlabel("Replies to Post Ratio (log scale)")
    plt.ylabel("Percentage of Appearance Related Replies (%)")
    plt.legend(title="Gender")
    plt.show()
    plt.savefig("scatter_reply_ratio_by_appearance.png")

def load_day_counts(original_df):
    original_df["day"] = original_df["created_at"].dt.date
    dates = original_df["day"].unique()
    if not os.path.exists("day_counts.csv"):
        records = []
        for date in dates:
            for gender in ["Male", "Female"]:
                n_replies = len(original_df[(original_df["day"] == date) & (original_df["gender"] == gender)])
                n_appearance_replies = original_df[(original_df["day"] == date) & (original_df["gender"] == gender) & (original_df["pred_contains_appearance"] == 1)].shape[0]
                proportion_appearance_replies = n_appearance_replies / n_replies if n_replies > 0 else np.nan
                records.append({
                    "day": date,
                    "gender": gender,
                    "total_replies": n_replies,
                    "appearance_replies": n_appearance_replies,
                })

        # Create dataframe of day by reply counts and appearance related reply counts
        day_counts = pd.DataFrame(records)
        day_counts.to_csv("day_counts.csv", index=False)

    else:
        day_counts = pd.read_csv("day_counts.csv")

    return day_counts 
    

def appearance_by_engagement_analysis(original_df, active_reply_counts):
    print("========== ENGAGEMENT LEVEL AND APPEARANCE RELATED REPLIES ==========")

    print(active_reply_counts["reply_to_post_ratio"].describe())

    correlation, p_value = stats.spearmanr(active_reply_counts["reply_to_post_ratio"].dropna(), active_reply_counts["proportion_appearance_replies"].dropna())
    print("Spearman's rank correlation between reply to post ratio and percentage of appearance related replies:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    women_reply_counts = active_reply_counts[active_reply_counts["gender"] == "Female"]
    correlation, p_value = stats.spearmanr(women_reply_counts["reply_to_post_ratio"].dropna(), women_reply_counts["proportion_appearance_replies"].dropna())
    print("Spearman's rank correlation between reply to post ratio and percentage of appearance related replies (Female):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    men_reply_counts = active_reply_counts[active_reply_counts["gender"] == "Male"]
    correlation, p_value = stats.spearmanr(men_reply_counts["reply_to_post_ratio"].dropna(), men_reply_counts["proportion_appearance_replies"].dropna())
    print("Spearman's rank correlation between reply to post ratio and percentage of appearance related replies (Male):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    day_counts = load_day_counts(original_df)

    print(day_counts["total_replies"].describe())

    day_counts["proportion_appearance_replies"] = day_counts["appearance_replies"] / day_counts["total_replies"]
    day_counts = day_counts[day_counts["total_replies"] > 20] # Filter out days with no replies to avoid division by zero

    # Scatter of proportion of appearance related replies by number of replies per day
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="total_replies", y="proportion_appearance_replies", data=day_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    plt.xscale("log")
    plt.title("Proportion of Appearance Related Replies by Number of Replies per Day")
    plt.xlabel("Number of Replies (log scale)")
    plt.ylabel("Proportion of Appearance Related Replies (%)")
    plt.show()
    plt.savefig("scatter_proportion_appearance_replies_by_reply_count.png")

    day_counts["total_replies"] = day_counts["total_replies"].replace(0, np.nan) # Replace 0 with NaN to avoid issues with logarithmic scale
    day_counts["log_reply_bins"] = pd.cut(day_counts["total_replies"], bins=np.logspace(np.log10(day_counts["total_replies"].min()), np.log10(day_counts["total_replies"].max()), 6, base=10), labels=np.logspace(np.log10(day_counts["total_replies"].min()), np.log10(day_counts["total_replies"].max()), 6, base=10)[1:])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="log_reply_bins", y="proportion_appearance_replies", data=day_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"}, ax=ax)
    plt.title("Proportion of Appearance Related Replies by Number of Replies per Day")
    plt.xlabel(f"Number of Replies (log scale from {day_counts['total_replies'].min()} to {day_counts['total_replies'].max()})")
    labels = range(1, 6)
    ax.set_xticklabels(labels, rotation=45)
    plt.ylabel("Proportion of Appearance Related Replies (%)")
    plt.show()
    plt.savefig("box_proportion_appearance_replies_by_reply_count.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="log_reply_bins", y="proportion_appearance_replies", data=day_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"}, ax=ax)
    plt.title("Proportion of Appearance Related Replies by Number of Replies per Day")
    plt.xlabel(f"Number of Replies (log scale from {day_counts['total_replies'].min()} to {day_counts['total_replies'].max()})")
    labels = range(1, 6)
    ax.set_xticklabels(labels, rotation=45)
    plt.ylabel("Proportion of Appearance Related Replies (%)")
    plt.show()
    plt.savefig("bar_proportion_appearance_replies_by_reply_count.png")


    #spearmans rank correlation between total_replies and proportion_appearance_replies
    correlation, p_value = stats.spearmanr(day_counts["total_replies"], day_counts["proportion_appearance_replies"],nan_policy='omit')
    print("Spearman's rank correlation between total replies and proportion of appearance related replies:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #spearmans rank correlation between total_replies and proportion_appearance_replies for women
    women_day_counts = day_counts[day_counts["gender"] == "Female"]
    correlation, p_value = stats.spearmanr(women_day_counts["total_replies"], women_day_counts["proportion_appearance_replies"],nan_policy='omit')
    print("Spearman's rank correlation between total replies and proportion of appearance related replies (Female):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #spearmans rank correlation between total_replies and proportion_appearance_replies for men 
    men_day_counts = day_counts[day_counts["gender"] == "Male"]
    correlation, p_value = stats.spearmanr(men_day_counts["total_replies"], men_day_counts["proportion_appearance_replies"],nan_policy='omit')
    print("Spearman's rank correlation between total replies and proportion of appearance related replies (Male):")
    print(f"Correlation: {correlation}, p-value: {p_value}")


    # Kendall's tau correlation between total_replies and proportion_appearance_replies
    correlation, p_value = stats.kendalltau(day_counts["total_replies"], day_counts["proportion_appearance_replies"],nan_policy='omit')
    print("Kendall's tau correlation between total replies and proportion of appearance related replies:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #Kendall's tau correlation between total_replies and proportion_appearance_replies for women
    correlation, p_value = stats.kendalltau(women_day_counts["total_replies"], women_day_counts["proportion_appearance_replies"],nan_policy='omit')
    print("Kendall's tau correlation between total replies and proportion of appearance related replies (Female):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #Kendall's tau correlation between total_replies and proportion_appearance_replies for men
    correlation, p_value = stats.kendalltau(men_day_counts["total_replies"], men_day_counts["proportion_appearance_replies"],nan_policy='omit')
    print("Kendall's tau correlation between total replies and proportion of appearance related replies (Male):")
    print(f"Correlation: {correlation}, p-value: {p_value}")


def simplify_ethnicity(original_df, mp_details, platform="bluesky"):
    print("======== SIMPLIFYING ETHNICITY CATEGORIES ========")
    print(f"Minority MPs where ethnicity is not known: {mp_details[(mp_details['minority_status'] == 'Minority') & (mp_details['ethnicity'].isna())]['mp_handle'].tolist()}")

    mp_details["ethnicity"] = mp_details["ethnicity"].fillna("Unknown")
    print(mp_details["ethnicity"].value_counts())
    print(f"Number of minority MPs: {mp_details[mp_details['ethnicity'] != 'Unknown'].shape[0]}")

    ethnicity_dict = {"British Indian": "Asian", "Black British": "Black", "British Pakistani": "Asian", "Black British/White British (Mixed)": "Mixed", "British Bangladeshi": "Asian", "British Chinese/White British (Mixed)": "Mixed", "British Chinese": "Asian", "British Bengali/White British (Mixed)": "Mixed", "British Palestinian/White British (Mixed)": "Mixed", "British Kurdish": "Other", "Anglo-Indian": "Mixed", "British Yemeni": "Other", "British Sri Lankan": "Asian", "British Syrian/White Irish (Mixed)": "Mixed", "British Cypriot": "Other", "Somali/White British (Mixed)": "Mixed", "British Indian": "Asian", "British Filipino/White British (Mixed)": "Mixed", "British Pakistani/Other White (Mixed)": "Mixed", "Unknown": "Unknown"}

    print(f"MPs missing category = {set(mp_details['ethnicity'].unique().tolist()) - set(ethnicity_dict.keys())}")

    mp_details["ethnicity_simplified"] = mp_details["ethnicity"].map(ethnicity_dict)

    original_df = original_df.merge(mp_details[["mp_handle", "ethnicity_simplified"]], on="mp_handle", how="left")
    print(len(original_df))
    print(original_df["ethnicity_simplified"].value_counts())
    print(original_df["ethnicity_simplified"].value_counts().sum())
    print(original_df["ethnicity_simplified"].isna().sum())

    return original_df, mp_details


def appearance_by_race_and_gender(original_df, active_reply_counts):
    print("======== APPEARANCE BY RACE AND GENDER ========")

    print("Proportion of replies which are appearance related by ethnicity and gender:")
    cross_tab = pd.crosstab(original_df["pred_contains_appearance"], [original_df["gender"], original_df["ethnicity_simplified"]], normalize="columns")
    print(cross_tab)

    #active_reply_counts["ethnicity"] = active_reply_counts["ethnicity"].fillna("Unknown")
    #active_reply_counts["ethnicity_simplified"] = active_reply_counts["ethnicity"].map(ethnicity_dict)
    print(active_reply_counts.groupby("gender")["ethnicity_simplified"].value_counts())

    mean_ratios = active_reply_counts.groupby("ethnicity_simplified")["reply_to_post_ratio"].mean()
    print(mean_ratios)

    print("Mean percentage of appearance related replies by ethnicity and gender:")
    means = active_reply_counts.groupby(["gender", "ethnicity_simplified"])["proportion_appearance_replies"].mean()
    print(means)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="reply_to_post_ratio", y="proportion_appearance_replies", data=active_reply_counts, hue="minority_status", style="gender")
    plt.xscale("log")
    plt.title("Percentage of Appearance Related Replies by Replies to Post Ratio, Colored by Ethnicity and Gender")
    plt.xlabel("Replies to Post Ratio (log scale)")
    plt.ylabel("Percentage of Appearance Related Replies (%)")
    plt.legend(title="Ethnicity and Gender")
    plt.show()
    #plt.savefig("scatter_reply_ratio_to_appearance_by_ethnicity.png")


    appearance_subcategories = original_df["pred_sub_category"].dropna().unique()
    for i, row in active_reply_counts.iterrows():
        for subcategory in appearance_subcategories:
            active_reply_counts.at[i, f"appearance_{subcategory}"] = original_df[f"appearance_{subcategory}"][original_df["mp_handle"] == row["mp_handle"]].sum()

    print(active_reply_counts.head(10))

    # Number of replies by MP ethnicity
    print(f"Number of replies by MP ethnicity:")
    print(original_df.groupby("ethnicity_simplified")["item_id"].nunique()/original_df["item_id"].nunique() * 100)

    # Number of appearance related replies by MP ethnicity
    print(f"Number of appearance related replies by MP ethnicity:")
    print(original_df[original_df["pred_contains_appearance"] == 1].groupby("ethnicity_simplified")["item_id"].nunique()/original_df[original_df["pred_contains_appearance"] == 1]["item_id"].nunique() * 100)

    for appearance_subcategory in appearance_subcategories:
        print(f"Number of replies which are {appearance_subcategory} by MP ethnicity:")
        print(original_df.groupby("ethnicity_simplified")[f"appearance_{appearance_subcategory}"].sum()/original_df[f"appearance_{appearance_subcategory}"].sum() * 100)

    # horizontal bar of proportion of appearance related replies which are in each subcategory by ethnicity
    subcategory_proportions = {}
    for ethnicity in original_df["ethnicity_simplified"].unique():
        subcategory_proportions[ethnicity] = {}
        for appearance_subcategory in appearance_subcategories:
            if appearance_subcategory == "none":
                continue
            subcategory_proportions[ethnicity][appearance_subcategory] = original_df[(original_df["ethnicity_simplified"] == ethnicity) & (original_df["pred_contains_appearance"] == 1)][f"appearance_{appearance_subcategory}"].sum()/len(original_df[(original_df["pred_contains_appearance"] == 1) & (original_df["ethnicity_simplified"] == ethnicity)]) * 100
    subcategory_proportions_df = pd.DataFrame(subcategory_proportions).transpose()
    fig, ax = plt.subplots(figsize=(18, 10))

    print(original_df[(original_df["appearance_facial_features"] == 1) & (original_df["ethnicity_simplified"] == "Black")]["cleaned_text"].sample(n=10).tolist())

    subcategory_proportions_df.plot(kind="barh", stacked=True, color=sns.color_palette("Set2", n_colors=len(appearance_subcategories)), ax=ax, legend=False)
    plt.title("Proportion of Appearance Related Replies in Each Subcategory by Ethnicity")
    plt.xlabel("Proportion of Appearance Related Replies by Subcategory (%)")
    plt.ylabel("Ethnicity")
    y_labels = [f"{ethnicity} (n = {original_df[original_df['ethnicity_simplified'] == ethnicity]['mp_handle'].nunique()}, r = {original_df[(original_df['ethnicity_simplified'] == ethnicity) & (original_df['pred_contains_appearance'] == 1)]['item_id'].nunique()})" for ethnicity in subcategory_proportions_df.index]
    plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), title="Appearance Subcategory")
    #plt.legend(title="Appearance Subcategory")
    plt.tight_layout()
    plt.show()
    #plt.savefig("stacked_bar_appearance_subcategories_by_ethnicity.png")

    return original_df, active_reply_counts


def appearance_by_age_and_gender(original_df, active_reply_counts):
    print("========== AGE AND APPEARANCE RELATED REPLIES ==========")

    # Scatter plot of percentage of appearance related replies by age, colored by gender
    plt.figure(figsize=(10, 6))
    active_reply_counts = active_reply_counts[active_reply_counts["wiki_birth_year"].notna()]
    active_reply_counts["age"] = 2026 - active_reply_counts["wiki_birth_year"].astype(int)
    print(active_reply_counts["age"].describe())

    # Correlation between age and percentage of appearance related replies
    correlation, p_value = stats.spearmanr(active_reply_counts["age"].dropna(), active_reply_counts["proportion_appearance_replies"].dropna())
    print("Spearman's rank correlation between age and percentage of appearance related replies:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    # Correlation between age and percentage of appearance related replies for women
    correlation_female, p_value_female = stats.spearmanr(active_reply_counts[active_reply_counts["gender"] == "Female"]["age"].dropna(), active_reply_counts[active_reply_counts["gender"] == "Female"]["proportion_appearance_replies"].dropna())
    print("Spearman's rank correlation between age and percentage of appearance related replies for women:")
    print(f"Correlation: {correlation_female}, p-value: {p_value_female}")

    # Correlation between age and percentage of appearance related replies for men
    correlation_male, p_value_male = stats.spearmanr(active_reply_counts[active_reply_counts["gender"] == "Male"]["age"].dropna(), active_reply_counts[active_reply_counts["gender"] == "Male"]["proportion_appearance_replies"].dropna())
    print("Spearman's rank correlation between age and percentage of appearance related replies for men:")
    print(f"Correlation: {correlation_male}, p-value: {p_value_male}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="age", y="proportion_appearance_replies", data=active_reply_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    #sns.regplot(x="age", y="proportion_appearance_replies", data=active_reply_counts[active_reply_counts["gender"] == "Male"], scatter=False, ax=plt.gca(), color="blue", label="Males")
    #sns.regplot(x="age", y="proportion_appearance_replies", data=active_reply_counts[active_reply_counts["gender"] == "Female"], scatter=False, ax=plt.gca(), color="lightblue", label="Females")
    plt.title("Percentage of Appearance Related Replies by Age")
    plt.xlabel("Age")
    plt.ylabel("Percentage of Appearance Related Replies (%)")
    plt.legend(title="Gender")
    plt.show()
    plt.savefig("scatter_age_by_appearance.png")

    plt.figure(figsize=(10, 6))
    print(f"Minimum age: {active_reply_counts['age'].min()}, Maximum age: {active_reply_counts['age'].max()}, Number of MPs with age data: {active_reply_counts['age'].notna().sum()}")
    # Mean percentage of appearance related replies by age group and gender
    active_reply_counts["age_group"] = pd.cut(active_reply_counts["age"], bins=[24, 35, 45, 55, 65, 100], labels=["24-34", "35-44", "45-54", "55-64", "65+"], include_lowest=True)
    # Number of MPs in each age group by gender
    print(active_reply_counts.groupby(["age_group", "gender"])["mp_handle"].nunique())
    mean_appearance_by_age_group = active_reply_counts.groupby(["age_group", "gender"])["proportion_appearance_replies"].mean().reset_index()
    print(mean_appearance_by_age_group)
    sns.barplot(x="age_group", y="proportion_appearance_replies", data=mean_appearance_by_age_group, hue="gender", palette={"Male": "blue", "Female": "lightblue"}, errorbar="sd")
    plt.title("Mean Percentage of Appearance Related Replies by Age Group and Gender")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Appearance Related Replies (%)")
    #plt.legend(title="Gender")
    plt.show()
    plt.savefig("bar_age_group_by_appearance.png")

    age_mapping = {24: "24-34", 25: "24-34", 26: "24-34", 27: "24-34", 28: "24-34", 29: "24-34", 30: "24-34", 31: "24-34", 32: "24-34", 33: "24-34", 34: "24-34",
                   35: "35-44", 36: "35-44", 37: "35-44", 38: "35-44", 39: "35-44", 40: "35-44", 41: "35-44", 42: "35-44", 43: "35-44", 44: "35-44",
                   45: "45-54", 46: "45-54", 47: "45-54", 48: "45-54", 49: "45-54", 50: "45-54", 51: "45-54", 52: "45-54", 53: "45-54", 54: "45-54",
                   55: "55-64", 56: "55-64", 57: "55-64", 58: "55-64", 59: "55-64", 60: "55-64", 61: "55-64", 62: "55-64", 63: "55-64", 64: "55-64",
                   65: "65+", 66: "65+", 67: "65+", 68: "65+", 69: "65+", 70: "65+", 71: "65+", 72: "65+", 73: "65+", 74: "65+", 75: "65+", 76: "65+", 77: "65+"}

    original_df["age"] = 2026 - original_df["wiki_birth_year"].astype(float, errors="ignore")
    original_df["age_group"] = original_df["age"].map(age_mapping)
    return original_df, active_reply_counts


def gbv_analysis(original_df, active_reply_counts):
    gbv_labels_raw_data = pd.read_csv("Bluesky_posts_RecollectionApr28_cleaned_gbv_multiclass_predictions.csv")
    print("GBV labels raw data loaded:")
    #print(gbv_labels_raw_data.columns)
    original_df = original_df.merge(gbv_labels_raw_data[["item_id", "combined_binary_pred", "combined_subtypes_pipe"]], on="item_id", how="left")
    print("GBV labels merged with original_df:")
    print(original_df.groupby("gender")["combined_binary_pred"].value_counts())
    cross_tab = pd.crosstab(original_df["combined_binary_pred"], original_df["gender"], normalize="columns")
    print("Cross tab of GBV labels by gender:")
    print(cross_tab)
    # Count of subtypes per gender
    subtypes = set(original_df["combined_subtypes_pipe"].dropna().str.split("|").explode())
    for subtype in subtypes:
        male = original_df[original_df["gender"] == "Male"]
        male_labels = male["combined_subtypes_pipe"].dropna().str.split("|").explode()
        male_count = Counter(male_labels)[subtype]
        female = original_df[original_df["gender"] == "Female"]
        female_labels = female["combined_subtypes_pipe"].dropna().str.split("|").explode()
        female_count = Counter(female_labels)[subtype]
        print(f"Proportion of GBV subtype {subtype} by gender:")
        print(f"For men: {male_count / male['item_id'].nunique() * 100 if male['item_id'].nunique() > 0 else 0}%")
        print(f"For women: {female_count / female['item_id'].nunique() * 100 if female['item_id'].nunique() > 0 else 0}%")

def main(args):
    if args.input_file:
        original_df = pd.read_csv(args.input_file)
    else:
        original_df = load_and_merge_data(platform=args.platform)

    rep_logger.info(f"Loaded data for platform: {args.platform}")

    # Filter data after 2024-07-04T00:00:00
    original_df["created_at"] = pd.to_datetime(original_df["created_at"], errors='coerce', utc=True, format="mixed")
    original_df = original_df[original_df["created_at"] >= pd.to_datetime("2024-07-04T00:00:00Z")]

    # Summary statistics before processing 
    #dataset_summary(original_df)

    # Load MP details and merge with original_df
    original_df, mp_details = load_mp_details_and_merge(original_df)

    # Analyse By Reply Type 
    #appearance_by_reply_type_analysis(original_df)

    # Analyse by post type (including duplicates)
    post_counts = mp_post_analysis(original_df)

    # Filtering to only public replies to MPs 
    original_df = original_df[original_df["item_type"] == "reply"]
    original_df = original_df[original_df["author_type"] == "by_public"]

    # Proportion of replies before deduplication:
    #proportion_replies_by_gender(original_df)

    original_df = original_df[original_df["duplicates"].isin(["first duplicate", "no duplicates"])]

    # Proportion of replies after deduplication:
    #proportion_replies_by_gender(original_df)

    # MPs with posts but no replies:
    mps_with_posts = set(post_counts["mp_handle"])
    mps_with_replies = set(original_df["mp_handle"])
    mps_with_posts_but_no_replies = mps_with_posts - mps_with_replies
    rep_logger.info(f"MPs with posts but no replies: {mps_with_posts_but_no_replies}")

    reply_counts = mp_reply_analysis(original_df)

    mp_details = mp_details[[f"{args.platform}_handle", "gender", "minority_status", "ethnicity", "wiki_birth_year"]]
    mp_details = pd.merge(mp_details, reply_counts[["mp_handle", "reply_count", "pred_contains_appearance"]], left_on=f"{args.platform}_handle", right_on="mp_handle", how="left")
    mp_details = pd.merge(mp_details, post_counts[["mp_handle", "post_count"]], left_on = f"{args.platform}_handle", right_on="mp_handle", how="left")
    rep_logger.info(f"MP details after merging: {mp_details.head()}")
    mp_details.rename(columns={f"{args.platform}_handle": "mp_handle"}, inplace=True)

    #plot as histogram
    #plot_reply_distribution(reply_counts)

    mp_details = mp_details[mp_details["post_count"].notna()] # Filter out MPs with no posts
    mp_details["active_filter"] = mp_details["reply_count"] > 20 # FILTERING OUT INACTIVE MPs AS THOSE WITH 20 OR FEWER REPLIES

    #compare_demographics_by_engagement_level(mp_details)

    active_mp_details = mp_details[mp_details["active_filter"] == True]
    active_mp_details["reply_to_post_ratio"] = active_mp_details["reply_count"] / active_mp_details["post_count"]
    active_mp_details["proportion_appearance_replies"] = active_mp_details.apply(lambda row: row["pred_contains_appearance"] / row["reply_count"] * 100 if row["reply_count"] > 0 else np.nan, axis=1)


    active_mps_list = active_mp_details["mp_handle"].tolist()
    print(f"Number of active MPs: {len(active_mps_list)}")

    original_df["active_filter"] = original_df["mp_handle"].isin(active_mps_list)
    active_df = original_df[original_df["active_filter"] == True]
    print("Number of replies to active MPs: ", len(active_df))

    assert active_mp_details["reply_count"].isna().sum() == 0, "There are MPs in active_reply_counts with no reply counts."
    assert active_mp_details["post_count"].isna().sum() == 0, "There are MPs in active_reply_counts with no post counts."
    assert len(active_mp_details["mp_handle"].unique()) == len(active_df["mp_handle"].unique()), "There are MPs in active_reply_counts which are not in active_df or vice versa."

    #mannwhitney_by_group(active_mp_details, label="reply_count", group_by="gender")
    #mannwhitney_by_group(active_mp_details, label="reply_to_post_ratio", group_by="gender")

    #dataset_summary(active_df)
    #proportion_replies_by_gender(active_mp_details)

    assert active_mp_details["proportion_appearance_replies"].isna().sum() == 0, "There are MPs in active_reply_counts with no proportion of appearance related replies."

    #mannwhitney_by_group(active_mp_details, label="proportion_appearance_replies", group_by="gender")


    # Appearance subtype analyses 
    #appearance_subtypes_by_gender(active_df)


    # One hot coding appearance subcategories
    appearance_subcategories = active_df["pred_sub_category"].dropna().unique()
    for subcategory in appearance_subcategories:
        active_df[f"appearance_{subcategory}"] = active_df["pred_sub_category"].apply(lambda x: 1 if x == subcategory else 0)

    appearance_by_engagement_analysis(original_df, active_mp_details)

    active_df, active_mp_details = simplify_ethnicity(active_df, active_mp_details)
    #active_df, active_mp_details = appearance_by_race_and_gender(active_df, active_mp_details)

    active_df, active_mp_details = appearance_by_age_and_gender(active_df, active_mp_details)

    gbv_analysis(active_df, active_mp_details)

    return active_df, active_mp_details

parser = argparse.ArgumentParser(description="Statistical analysis of appearance related replies to MPs on Bluesky")
parser.add_argument("--input_file", type=str, default="bluesky_full_info_RecollectionApr28.csv", help="Path to the input CSV file")
parser.add_argument("--platform", type=str, default="bluesky", help="Platform to analyze (default: bluesky)")
args = parser.parse_args()


if __name__ == "__main__":
    prepped_df, prepped_mp_details = main(args)
    

    #prepped_df.to_csv("original_df_with_demographics.csv", index=False)


    exit()

# To Do: Sentiment, Relationship between google trends and appearance related replies









