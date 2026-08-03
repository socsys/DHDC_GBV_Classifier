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

import statsmodels.formula.api as smf


# Logger for report
rep_logger = logging.getLogger("report_logger")
rep_logger.addHandler(logging.FileHandler("report.log", mode="w"))

# Logger for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='analysis.log', filemode='w')

matplotlib.rc({'font.size': 14})
plt.rcParams.update({'font.size': 14})

def load_and_merge_data(platform="bluesky", new=False):
    ''' Function to load and merge default labelled data and raw data, and save merged data as csv for future use to avoid having to merge every time. If merged csv already exists, load that instead.'''

    if platform == "bluesky":
        if new==True:
            raw_df = pd.read_csv("bluesky_posts_currentMPs_Jun_cleaned.csv")
            labelled_df = pd.read_csv("bluesky_posts_currentMPs_Jun_cleaned_predictions.csv")
            original_df = raw_df.merge(labelled_df[["data_id","pred_binary", "pred_category_pipe"]], left_on="item_id", right_on="data_id", how="left")

            print(len(original_df))

            original_df["created_at"] = original_df["created_at"].apply(lambda x: pd.to_datetime(x, errors='coerce')) # Force created_at to be datetime, coercing errors to NaT

            original_df.to_csv("bluesky_posts_currentMPs_Jun_full.csv", index=False)

        else:
            original_df = pd.read_csv("bluesky_posts_currentMPs_Jun_full.csv")
    elif platform == "twitter":
        pass # To Do: Implement for X/Twitter
    else:
        raise ValueError("Platform not supported. Please choose 'bluesky' or 'twitter'.")
    
    return original_df

def dataset_summary(df):
    ''' Function to compute summary statistics for the dataset. '''
    print("======== DATASET SUMMARY ========")
    print(f"Total number of observations: {len(df)}")
    print(f"Number of unique MPs: {df['mp_handle'].nunique()}")
    print(f"Number of MPs by gender: {df.groupby('gender')['mp_handle'].nunique()}")
    print(f"Number of unique authors: {df['author'].nunique()}")

def load_mp_details_and_merge(original_df):
    print("======== LOADING MP DETAILS ========")
    mp_details = pd.read_csv("FINAL-mp-details.csv", dtype={"theyworkforyou_id": str, "parliament_id": str, "wiki_birth_year": str})
    mp_details = mp_details.replace("—", np.nan)
    original_df = original_df.merge(mp_details[["bluesky_handle", "minority_status", "ethnicity", "party", "wiki_birth_year"]], left_on="mp_handle", right_on="bluesky_handle", how="left")
    print(f"Number of MPs with bluesky_handle in mp_details: {mp_details['bluesky_handle'].nunique()}")
    return original_df, mp_details

def mp_post_analysis(original_df):
    print("======== MP POST ANALYSIS ========")
    posts_only = original_df[original_df["item_type"] == "post"]
    print(f"Number of MPs with posts: {posts_only['mp_handle'].nunique()}")
    print(f"Number of MPs with posts by gender: {posts_only.groupby('gender')['mp_handle'].nunique()}")
    post_counts = posts_only.groupby("mp_handle").size().reset_index(name="post_count")
    print(post_counts["post_count"].median())
    print("Number of posts by duplicate status:")
    print(posts_only.groupby("duplicates")["item_id"].count())
    print("Number of posts by gender:")
    print(posts_only.groupby("gender")["item_id"].count())
    print("Reply to post ratio by gender:")
    reply_counts = original_df[original_df["item_type"].isin(["reply-1", "reply-2"])].groupby("gender")["item_id"].count()
    print(reply_counts / posts_only.groupby("gender")["item_id"].count())
    return post_counts

def proportion_replies_by_gender(original_df):
    print("======== PROPORTION OF REPLIES BY GENDER ========")
    print(f"Length of dataset: {len(original_df)}")
    print("Proportion of replies by gender:")
    replies_to_women = original_df[original_df["gender"] == "Female"]
    print("Women:", len(replies_to_women)/len(original_df) * 100)
    print("Male:", len(original_df[original_df["gender"] == "Male"])/len(original_df) * 100)
    print("Number of replies by gender:")
    print(original_df[original_df["item_type"].isin(["reply-1", "reply-2"])].groupby("gender")["item_id"].count())

def duplicates_analysis(original_df, label="pred_contains_appearance"):
    print("======== DUPLICATES ANALYSIS ========")
    print(f"Proportion of {label} in replies by duplicates:")
    print(original_df.groupby("duplicates")[label].mean()*100)
    print(f"Proportion of {label} comments in replies by duplicates:")
    print(original_df.groupby("duplicates")[label].mean()*100)

def mp_reply_analysis(original_df):
    ''' Function to analyze replies to MPs '''
    print("======== MP REPLY ANALYSIS ========")
    assert all(original_df["item_type"].isin(["reply-1", "reply-2"])), "Not all items in original_df are replies."
    print(f"Number of MPs with replies: {original_df['mp_handle'].nunique()}")
    print(f"Number of MPs with replies by gender: {original_df.groupby('gender')['mp_handle'].nunique()}")
    reply_counts = original_df.groupby("mp_handle").size().reset_index(name="reply_count")
    #reply_counts["pred_contains_appearance"] = original_df.groupby("mp_handle")["pred_contains_appearance"].sum().values
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
    print(reply_counts.groupby("gender")["percentage_appearance_replies"].mean())
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
    sns.scatterplot(x="reply_to_post_ratio", y="percentage_appearance_replies", data=active_reply_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    # x scale logarithmic
    plt.xscale("log")
    plt.title("Percentage of Appearance Related Replies by Replies to Post Ratio")
    plt.xlabel("Replies to Post Ratio (log scale)")
    plt.ylabel("Percentage of Appearance Related Replies (%)")
    plt.legend(title="Gender")
    plt.show()
    plt.savefig("scatter_reply_ratio_by_appearance.png")

def load_day_counts(original_df, new=False):
    original_df["day"] = original_df["created_at"].dt.date
    dates = original_df["day"].unique()
    if not os.path.exists("day_counts.csv") or new==True:
        records = []
        for date in dates:
            for gender in ["Male", "Female"]:
                n_replies = len(original_df[(original_df["day"] == date) & (original_df["gender"] == gender)])
                n_appearance_replies = original_df[(original_df["day"] == date) & (original_df["gender"] == gender) & (original_df["pred_contains_appearance"] == 1)].shape[0]
                percentage_appearance_replies = n_appearance_replies / n_replies if n_replies > 0 else np.nan
                n_gbv_replies = original_df[(original_df["day"] == date) & (original_df["gender"] == gender) & (original_df["pred_binary"] == 1)].shape[0]
                records.append({
                    "day": date,
                    "gender": gender,
                    "total_replies": n_replies,
                    "appearance_replies": n_appearance_replies,
                    "gbv_replies": n_gbv_replies
                })

        # Create dataframe of day by reply counts and appearance related reply counts
        day_counts = pd.DataFrame(records)
        day_counts.to_csv("day_counts.csv", index=False)

    else:
        day_counts = pd.read_csv("day_counts.csv")

    return day_counts 

def plot_appearance_by_engagement(day_counts, chart_type, label="percentage_appearance_replies"):
    ''' Function to visualize proportion of appearance related replies by number of replies per day '''
    plt.figure(figsize=(10, 6))
    if label == "percentage_appearance_replies":
        plt.title("Percentage of Appearance Related Replies by Number of Replies per Day")
        plt.ylabel("Percentage of Appearance Related Replies (%)")
    elif label == "percentage_gbv_replies":
        plt.title("Percentage of GBV Related Replies by Number of Replies per Day")
        plt.ylabel("Percentage of GBV Related Replies (%)")

    if chart_type == "scatter":
        sns.scatterplot(x="total_replies", y=label, data=day_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
        plt.xscale("log")
        plt.xlabel("Number of Replies (log scale)")
        
    if chart_type == "box":
        sns.boxplot(x="log_reply_bins", y=label, data=day_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
        plt.xlabel(f"Number of Replies (log scale from {day_counts['total_replies'].min()} to {day_counts['total_replies'].max()})")
        labels = range(1, 6)
        plt.xticks(labels, rotation=45)

    if chart_type == "bar":
        sns.barplot(x="log_reply_bins", y=label, data=day_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
        plt.xlabel(f"Number of Replies (log scale from {day_counts['total_replies'].min()} to {day_counts['total_replies'].max()})")
        labels = range(1, 6)
        plt.xticks(labels, rotation=45)
    
    plt.show()
    plt.savefig(f"{chart_type}_{label}_by_reply_count.png")


def log_reply_bins(day_counts):
    ''' Function to create logarithmic bins for number of replies per day '''
    day_counts["total_replies"] = day_counts["total_replies"].replace(0, np.nan) # Replace 0 with NaN to avoid issues with logarithmic scale
    day_counts["log_reply_bins"] = pd.cut(day_counts["total_replies"], bins=np.logspace(np.log10(day_counts["total_replies"].min()), np.log10(day_counts["total_replies"].max()), 6, base=10), labels=np.logspace(np.log10(day_counts["total_replies"].min()), np.log10(day_counts["total_replies"].max()), 6, base=10)[1:])
    return day_counts
    

def appearance_by_engagement_analysis(original_df, active_reply_counts, label="percentage_appearance_replies"):
    print(f"========== ENGAGEMENT LEVEL AND {label.upper()} RELATED REPLIES ==========")

    print(active_reply_counts["reply_to_post_ratio"].describe())

    correlation, p_value = stats.spearmanr(active_reply_counts["reply_to_post_ratio"].dropna(), active_reply_counts[label].dropna())
    print(f"Spearman's rank correlation between reply to post ratio and {label.replace('_', ' ')}:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    women_reply_counts = active_reply_counts[active_reply_counts["gender"] == "Female"]
    correlation, p_value = stats.spearmanr(women_reply_counts["reply_to_post_ratio"].dropna(), women_reply_counts[label].dropna())
    print(f"Spearman's rank correlation between reply to post ratio and {label.replace('_', ' ')} (Women):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    men_reply_counts = active_reply_counts[active_reply_counts["gender"] == "Male"]
    correlation, p_value = stats.spearmanr(men_reply_counts["reply_to_post_ratio"].dropna(), men_reply_counts[label].dropna())
    print(f"Spearman's rank correlation between reply to post ratio and {label.replace('_', ' ')} (Male):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    # scatter of reply to post ratio by labeled replies coloured by gender
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="reply_to_post_ratio", y=label, data=active_reply_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    plt.title(f"{label.replace('_', ' ').title()} by Reply to Post Ratio.")
    plt.xscale("log")
    plt.xlabel("Reply to Post Ratio")
    plt.ylabel(label.replace('_', ' ').title())
    plt.show()
    plt.savefig(f"Scatter_{label}_by_reply_to_post_ratio.png")

    day_counts = load_day_counts(original_df, new=True) 

    print(day_counts["total_replies"].describe())

    day_counts["percentage_appearance_replies"] = day_counts["appearance_replies"] / day_counts["total_replies"] * 100
    day_counts["percentage_gbv_replies"] = day_counts["gbv_replies"] / day_counts["total_replies"] * 100
    day_counts = day_counts[day_counts["total_replies"] > 20] # Filter out days with no replies to avoid division by zero

    # Scatter of proportion of appearance related replies by number of replies per day
    plot_appearance_by_engagement(day_counts, "scatter", label=label)

    day_counts = log_reply_bins(day_counts)
    plot_appearance_by_engagement(day_counts, "box", label=label)
    plot_appearance_by_engagement(day_counts, "bar", label=label)


    #spearmans rank correlation between total_replies and percentage_appearance_replies
    correlation, p_value = stats.spearmanr(day_counts["total_replies"], day_counts[label],nan_policy='omit')
    print(f"Spearman's rank correlation between total replies and {label.replace('_', ' ')}:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #spearmans rank correlation between total_replies and percentage_appearance_replies for women
    women_day_counts = day_counts[day_counts["gender"] == "Female"]
    correlation, p_value = stats.spearmanr(women_day_counts["total_replies"], women_day_counts[label],nan_policy='omit')
    print(f"Spearman's rank correlation between total replies and {label.replace('_', ' ')} (Women):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #spearmans rank correlation between total_replies and percentage_appearance_replies for men
    men_day_counts = day_counts[day_counts["gender"] == "Male"]
    correlation, p_value = stats.spearmanr(men_day_counts["total_replies"], men_day_counts[label],nan_policy='omit')
    print(f"Spearman's rank correlation between total replies and {label.replace('_', ' ')} (Male):")
    print(f"Correlation: {correlation}, p-value: {p_value}")


    # Kendall's tau correlation between total_replies and percentage_appearance_replies
    correlation, p_value = stats.kendalltau(day_counts["total_replies"], day_counts[label],nan_policy='omit')
    print(f"Kendall's tau correlation between total replies and {label.replace('_', ' ')}:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #Kendall's tau correlation between total_replies and percentage_appearance_replies for women
    correlation, p_value = stats.kendalltau(women_day_counts["total_replies"], women_day_counts[label],nan_policy='omit')
    print(f"Kendall's tau correlation between total replies and {label.replace('_', ' ')} (Women):")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    #Kendall's tau correlation between total_replies and percentage_appearance_replies for men
    correlation, p_value = stats.kendalltau(men_day_counts["total_replies"], men_day_counts[label],nan_policy='omit')
    print(f"Kendall's tau correlation between total replies and {label.replace('_', ' ')} (Male):")
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
    assert len(original_df) == original_df["ethnicity_simplified"].value_counts().sum(),  "Some data points missing ethnicity information after merging with mp_details. Check for missing values in mp_details or original_df."
    print(original_df["ethnicity_simplified"].value_counts())
    print(f"Number of MPs with unknown ethnicity: {original_df['ethnicity_simplified'].isna().sum()}")

    return original_df, mp_details

def gbv_subcategories_analysis(original_df, gbv_subcategories):
    # horizontal bar of proportion of gbv related replies which are in each subcategory by ethnicity
    subcategory_proportions = {}
    for ethnicity in original_df["ethnicity_simplified"].unique():
        subcategory_proportions[ethnicity] = {}
        for gbv_subcategory in gbv_subcategories:
            if gbv_subcategory == "none":
                continue
            subcategory_proportions[ethnicity][gbv_subcategory] = original_df[(original_df["ethnicity_simplified"] == ethnicity) & (original_df["pred_binary"] == 1)][f"gbv_{gbv_subcategory}"].sum()/len(original_df[(original_df["pred_binary"] == 1) & (original_df["ethnicity_simplified"] == ethnicity)]) * 100
    subcategory_proportions_df = pd.DataFrame(subcategory_proportions).transpose()
    fig, ax = plt.subplots(figsize=(18, 10))

    subcategory_proportions_df.plot(kind="barh", stacked=True, color=sns.color_palette("Set2", n_colors=len(gbv_subcategories)), ax=ax, legend=False)
    plt.title("Proportion of GBV Related Replies in Each Subcategory by Ethnicity")
    plt.xlabel("Proportion of GBV Related Replies by Subcategory (%)")
    plt.ylabel("Ethnicity")
    y_labels = [f"{ethnicity} (n = {original_df[original_df['ethnicity_simplified'] == ethnicity]['mp_handle'].nunique()}, r = {original_df[(original_df['ethnicity_simplified'] == ethnicity) & (original_df['pred_binary'] == 1)]['item_id'].nunique()})" for ethnicity in subcategory_proportions_df.index]
    plt.yticks(ticks=range(len(y_labels)), labels=y_labels)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), title="GBV Subcategory")
    #plt.legend(title="Appearance Subcategory")
    plt.tight_layout()
    plt.show()
    #plt.savefig("stacked_bar_appearance_subcategories_by_ethnicity.png")


def appearance_by_race_and_gender(original_df, active_reply_counts, label="pred_binary"):
    print(f"======== {label.replace('_', ' ').upper()} BY RACE AND GENDER ========")

    print(f"{label.replace('_', ' ').upper()} by ethnicity and gender:")
    cross_tab = pd.crosstab(original_df[label], [original_df["gender"], original_df["ethnicity_simplified"]], normalize="columns")
    print(cross_tab)

    #active_reply_counts["ethnicity"] = active_reply_counts["ethnicity"].fillna("Unknown")
    #active_reply_counts["ethnicity_simplified"] = active_reply_counts["ethnicity"].map(ethnicity_dict)
    if label == "pred_binary":
        percentage_label = "percentage_gbv_replies"
    elif label == "pred_contains_appearance":
        percentage_label = "percentage_appearance_replies"
    print(active_reply_counts.groupby("gender")["ethnicity_simplified"].value_counts())

    mean_ratios = active_reply_counts.groupby("ethnicity_simplified")["reply_to_post_ratio"].mean()
    print(mean_ratios)

    print(f"Mean {label.replace('_', ' ').upper()} by ethnicity and gender:")
    means = active_reply_counts.groupby(["gender", "ethnicity_simplified"])[percentage_label].mean()
    print(means)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="reply_to_post_ratio", y=percentage_label, data=active_reply_counts, hue="minority_status", style="gender")
    plt.xscale("log")
    plt.title(f"{label.replace('_', ' ').upper()} by Replies to Post Ratio, Colored by Ethnicity and Gender")
    plt.xlabel("Replies to Post Ratio (log scale)")
    plt.ylabel("Percentage of Appearance Related Replies (%)")
    plt.legend(title="Ethnicity and Gender")
    plt.show()
    plt.savefig(f"scatter_reply_ratio_to_{percentage_label}_by_ethnicity.png")

    #print(active_reply_counts.head(10))

    ## Number of replies by MP ethnicity
    print(f"Number of replies by MP ethnicity:")
    print(original_df.groupby("ethnicity_simplified")["item_id"].nunique()/original_df["item_id"].nunique() * 100)

    ## Number of target replies by MP ethnicity
    print(f"Number of target replies by MP ethnicity:")
    print(original_df[original_df[label] == 1].groupby("ethnicity_simplified")["item_id"].nunique()/original_df[original_df[label] == 1]["item_id"].nunique() * 100)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="ethnicity_simplified", y=percentage_label, data=active_reply_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    if percentage_label == "percentage_appearance_replies":
        plt.title(f"Percentage of Appearance Related Replies by Ethnicity and Gender")
        plt.ylabel(f"Percentage of Appearance Related Replies (%)")
    elif percentage_label == "percentage_gbv_replies":
        plt.title(f"Percentage of GBV Related Replies by Ethnicity and Gender")
        plt.ylabel(f"Percentage of GBV Related Replies (%)")
    plt.xlabel("Ethnicity")
    plt.legend(title="Gender")
    plt.show()
    plt.savefig(f"box_{percentage_label}_by_ethnicity_and_gender.png")

    ## Linear regression of percentage of appearance related replies by ethnicity and gender
    linear_model = smf.ols(formula=f"{percentage_label} ~ C(minority_status) + C(gender) + C(minority_status):C(gender)", data=active_reply_counts).fit()
    print(linear_model.summary())

    mannwhitney_by_group(active_reply_counts, label=percentage_label, group_by="minority_status")

def handle_age_data(active_df, active_mp_details):
    print("======== HANDLING AGE DATA ========")
    # Convert wiki_birth_year to numeric, coercing errors to NaN
    active_mp_details["wiki_birth_year"] = pd.to_numeric(active_mp_details["wiki_birth_year"], errors="coerce")
    active_df["wiki_birth_year"] = pd.to_numeric(active_df["wiki_birth_year"], errors="coerce")

    age_mapping = {24: "24-34", 25: "24-34", 26: "24-34", 27: "24-34", 28: "24-34", 29: "24-34", 30: "24-34", 31: "24-34", 32: "24-34", 33: "24-34", 34: "24-34",
                   35: "35-44", 36: "35-44", 37: "35-44", 38: "35-44", 39: "35-44", 40: "35-44", 41: "35-44", 42: "35-44", 43: "35-44", 44: "35-44",
                   45: "45-54", 46: "45-54", 47: "45-54", 48: "45-54", 49: "45-54", 50: "45-54", 51: "45-54", 52: "45-54", 53: "45-54", 54: "45-54",
                   55: "55-64", 56: "55-64", 57: "55-64", 58: "55-64", 59: "55-64", 60: "55-64", 61: "55-64", 62: "55-64", 63: "55-64", 64: "55-64",
                   65: "65+", 66: "65+", 67: "65+", 68: "65+", 69: "65+", 70: "65+", 71: "65+", 72: "65+", 73: "65+", 74: "65+", 75: "65+", 76: "65+", 77: "65+"}

    active_mp_details["age_group"] = active_mp_details["wiki_birth_year"].apply(lambda x: age_mapping.get(2026 - x, "Unknown") if pd.notna(x) else "Unknown")
    active_df["age_group"] = active_df["wiki_birth_year"].apply(lambda x: age_mapping.get(2026 - x, "Unknown") if pd.notna(x) else "Unknown")

    # display all pandas columns in the output
    pd.set_option('display.max_columns', None)

    print(active_mp_details[["mp_handle", "wiki_birth_year","age_group"]].head())
    print(active_df[["mp_handle", "cleaned_text", "wiki_birth_year","age_group", "pred_binary", "pred_contains_appearance"]].head())

    return active_df, active_mp_details


def appearance_by_age_and_gender(original_df, active_mp_details, label="percentage_appearance_replies"):
    print(f"========== AGE AND {label.replace('_', ' ').upper()} ==========")

    active_reply_counts = active_reply_counts[active_reply_counts["wiki_birth_year"].notna()]
    active_reply_counts["age"] = 2026 - active_reply_counts["wiki_birth_year"].astype(int)
    print(active_reply_counts["age"].describe())

    # Correlation between age and percentage of appearance related replies
    correlation, p_value = stats.spearmanr(active_reply_counts["age"].dropna(), active_reply_counts[label].dropna())
    print("Spearman's rank correlation between age and percentage of appearance related replies:")
    print(f"Correlation: {correlation}, p-value: {p_value}")

    # Correlation between age and percentage of appearance related replies for women
    correlation_female, p_value_female = stats.spearmanr(active_reply_counts[active_reply_counts["gender"] == "Female"]["age"].dropna(), active_reply_counts[active_reply_counts["gender"] == "Female"][label].dropna())
    print("Spearman's rank correlation between age and percentage of appearance related replies for women:")
    print(f"Correlation: {correlation_female}, p-value: {p_value_female}")

    # Correlation between age and percentage of appearance related replies for men
    correlation_male, p_value_male = stats.spearmanr(active_reply_counts[active_reply_counts["gender"] == "Male"]["age"].dropna(), active_reply_counts[active_reply_counts["gender"] == "Male"][label].dropna())
    print("Spearman's rank correlation between age and percentage of appearance related replies for men:")
    print(f"Correlation: {correlation_male}, p-value: {p_value_male}")

    # Scatter plot of percentage of appearance related replies by age, colored by gender
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="age", y=label, data=active_reply_counts, hue="gender", palette={"Male": "blue", "Female": "lightblue"})
    #sns.regplot(x="age", y="percentage_appearance_replies", data=active_reply_counts[active_reply_counts["gender"] == "Male"], scatter=False, ax=plt.gca(), color="blue", label="Males")
    #sns.regplot(x="age", y="percentage_appearance_replies", data=active_reply_counts[active_reply_counts["gender"] == "Female"], scatter=False, ax=plt.gca(), color="lightblue", label="Females")
    plt.title(f"{label.replace('_', ' ').title()} by Age")
    plt.xlabel("Age")
    plt.ylabel(f"Percentage of {label.replace('_', ' ').title()} (%)")
    plt.legend(title="Gender")
    plt.show()
    plt.savefig(f"scatter_age_by_{label}.png")

    plt.figure(figsize=(10, 6))
    print(f"Minimum age: {active_reply_counts['age'].min()}, Maximum age: {active_reply_counts['age'].max()}, Number of MPs with age data: {active_reply_counts['age'].notna().sum()}")
    # Mean percentage of appearance related replies by age group and gender
    active_reply_counts["age_group"] = pd.cut(active_reply_counts["age"], bins=[24, 35, 45, 55, 65, 100], labels=["24-34", "35-44", "45-54", "55-64", "65+"], include_lowest=True)
    # Number of MPs in each age group by gender
    print(active_reply_counts.groupby(["age_group", "gender"])["mp_handle"].nunique())
    mean_appearance_by_age_group = active_reply_counts.groupby(["age_group", "gender"])[label].mean().reset_index()
    print(mean_appearance_by_age_group)
    sns.barplot(x="age_group", y=label, data=mean_appearance_by_age_group, hue="gender", palette={"Male": "blue", "Female": "lightblue"}, errorbar="sd")
    plt.title(f"{label.replace('_', ' ').title()} by Age Group and Gender")
    plt.xlabel("Age Group")
    plt.ylabel(f"Mean {label.replace('_', ' ').title()} (%)")
    #plt.legend(title="Gender")
    plt.show()
    plt.savefig(f"bar_age_group_by_{label}.png")

def gbv_analysis(original_df, active_reply_counts):
    gbv_labels_raw_data = pd.read_csv("inference_predictions.csv")
    print("GBV labels raw data loaded:")
    #print(gbv_labels_raw_data.columns)
    original_df = original_df.merge(gbv_labels_raw_data[["comment_id", "pred_binary", "pred_category_pipe"]], left_on="item_id", right_on="comment_id", how="left")
    print("GBV labels merged with original_df:")
    print(original_df.groupby("gender")["pred_binary"].value_counts())
    cross_tab = pd.crosstab(original_df["pred_binary"], original_df["gender"], normalize="columns")
    print("Cross tab of GBV labels by gender:")
    print(cross_tab)
    # Count of subtypes per gender
    subtypes = set(original_df["pred_category_pipe"].dropna().str.split("|").explode())
    for subtype in subtypes:
        male = original_df[original_df["gender"] == "Male"]
        male_labels = male["pred_category_pipe"].dropna().str.split("|").explode()
        male_count = Counter(male_labels)[subtype]
        female = original_df[original_df["gender"] == "Female"]
        female_labels = female["pred_category_pipe"].dropna().str.split("|").explode()
        female_count = Counter(female_labels)[subtype]
        print(f"Proportion of GBV subtype {subtype} by gender:")
        print(f"For men: {male_count / male['item_id'].nunique() * 100 if male['item_id'].nunique() > 0 else 0}%")
        print(f"For women: {female_count / female['item_id'].nunique() * 100 if female['item_id'].nunique() > 0 else 0}%")

def select_active_mps(mp_details, original_df, reply_counts):
    print("======== SELECTING ACTIVE MPs ========")
    mp_details = mp_details[mp_details["post_count"].notna()] # Filter out MPs with no posts
    mp_details["active_filter"] = mp_details["reply_count"] >= 20 # FILTERING OUT INACTIVE MPs AS THOSE WITH 20 OR FEWER REPLIES

    compare_demographics_by_engagement_level(mp_details)

    active_mp_details = mp_details[mp_details["active_filter"] == True]
    active_mp_details["reply_to_post_ratio"] = active_mp_details["reply_count"] / active_mp_details["post_count"]

    active_mps_list = active_mp_details["mp_handle"].tolist()
    print(f"Number of active MPs: {len(active_mps_list)}")
    print(f"Number of active MPs by gender: {active_mp_details.groupby('gender')['mp_handle'].nunique()}")

    original_df["active_filter"] = original_df["mp_handle"].isin(active_mps_list)
    active_df = original_df[original_df["active_filter"] == True]
    print("Number of replies to active MPs: ", len(active_df))
    
    assert active_mp_details["reply_count"].isna().sum() == 0, "There are MPs in active_reply_counts with no reply counts."
    assert active_mp_details["post_count"].isna().sum() == 0, "There are MPs in active_reply_counts with no post counts."
    assert len(active_mp_details["mp_handle"].unique()) == len(active_df["mp_handle"].unique()), "There are MPs in active_reply_counts which are not in active_df or vice versa."

    return active_df, active_mp_details

def proportion_target_replies_by_gender(active_df, label="pred_binary"):
    print(f"======== TARGET REPLIES BY GENDER ({label}) ========")
    print(f"Number of replies which are {label}: {active_df[label].sum()}")
    print(f"Number of replies which are {label} by gender:")
    cross_tab = pd.crosstab(active_df[label], active_df["gender"])
    print(cross_tab)
    print(f"Proportion of replies which are {label} by gender:")
    cross_tab = pd.crosstab(active_df[label], active_df["gender"], normalize="columns")
    print(cross_tab)

def main(args):
    if args.input_file:
        original_df = pd.read_csv(args.input_file)
    else:
        original_df = load_and_merge_data(platform=args.platform, new=args.new)

    # Load and merge GBV labels 


    rep_logger.info(f"Loaded data for platform: {args.platform}")

    # Filter data after 2024-07-04T00:00:00
    original_df["created_at"] = pd.to_datetime(original_df["created_at"], errors='coerce', utc=True, format="mixed")
    original_df = original_df[(original_df["created_at"] >= pd.to_datetime("2026-01-01T00:00:00Z"))&(original_df["created_at"] <= pd.to_datetime("2026-06-30T23:59:59Z"))]

    ## Summary statistics before processing 
    #dataset_summary(original_df)

    ## Load MP details and merge with original_df
    original_df, mp_details = load_mp_details_and_merge(original_df)


    ## Analyse By Reply Type 
    #appearance_by_reply_type_analysis(original_df)

    ## Analyse by post type (including duplicates)
    #mp_post_analysis(original_df)
    #mp_replies_analysis(original_df)

    ## Export list of MPs who have bluesky account but did not post during the period of analysis
    mps_with_posts = set(original_df["mp_handle"])
    mps_with_no_posts = set(mp_details[f"{args.platform}_handle"]) - mps_with_posts
    rep_logger.info(f"MPs with no posts during the period of analysis: {mps_with_no_posts}")
    with open(f"mps_with_no_posts_{args.platform}.txt", "w") as f:
        for mp in mps_with_no_posts:
            f.write(f"{mp}\n")

    original_df = original_df[original_df["duplicates"].isin(["first duplicate", "no duplicates"])]

    # Analyse by post type (excluding duplicates)
    post_counts = mp_post_analysis(original_df)

    # Filtering to only public replies to MPs 
    original_df = original_df[(original_df["item_type"] == "reply-1") | (original_df["item_type"] == "reply-2")]
    original_df = original_df[original_df["author_type"] == "by_public"]

    ## Proportion of replies before deduplication:
    #proportion_replies_by_gender(original_df)

    ## Proportion of replies after deduplication:
    #proportion_replies_by_gender(original_df)


    ## MPs with posts but no replies:
    mps_with_posts = set(post_counts["mp_handle"])
    mps_with_replies = set(original_df["mp_handle"])
    mps_with_posts_but_no_replies = mps_with_posts - mps_with_replies
    rep_logger.info(f"MPs with posts but no replies: {mps_with_posts_but_no_replies}")

    reply_counts = mp_reply_analysis(original_df)

    print(mp_details.columns)

    mp_details = mp_details[["parliament_name",
    f"{args.platform}_handle", "gender", "minority_status", "ethnicity", "wiki_birth_year"]]
    mp_details.dropna(subset=[f"{args.platform}_handle"], inplace=True)
    print("Number of MPs with handles after dropping NaNs: ", len(mp_details))
    mp_details = pd.merge(mp_details, reply_counts[["mp_handle", "reply_count"]], left_on=f"{args.platform}_handle", right_on="mp_handle", how="left")
    mp_details = pd.merge(mp_details, post_counts[["mp_handle", "post_count"]], left_on = f"{args.platform}_handle", right_on="mp_handle", how="left")
    rep_logger.info(f"MP details after merging: {mp_details.head()}")
    mp_details.rename(columns={f"{args.platform}_handle": "mp_handle"}, inplace=True)

    mp_details.drop(columns=["mp_handle_y", "mp_handle_x"], inplace=True)
    mp_details = mp_details.apply(lambda x: x.fillna(0) if x.name in ["reply_count", "post_count"] else x)

    print("MP details after merging with reply and post counts: ", mp_details.head())

    ## plot as histogram
    #plot_reply_distribution(reply_counts)


    active_df, active_mp_details = select_active_mps(mp_details, original_df, reply_counts)
    print(f"Length of active_df: {len(active_df)}, Length of active_mp_details: {len(active_mp_details)}")

    print(f"Number of replies which are NaN: {active_df['pure_text'].isna().sum()}")
    print(f"Number of replies which are empty strings: {(active_df['pure_text'] == '').sum()}")

    #mannwhitney_by_group(active_mp_details, label="reply_count", group_by="gender")
    #mannwhitney_by_group(active_mp_details, label="reply_to_post_ratio", group_by="gender")

    ## Incorporate appearance related labels into active_df and active_mp_details
    appearance_df = pd.read_csv("bluesky_replies_jan-jun_for_analysis.csv")
    assert len(appearance_df) == len(active_df), "The length of appearance_df does not match the length of active df for analysis. Please check the data."

    active_df = active_df.merge(appearance_df[["item_id", "pred_contains_appearance"]], on="item_id", how="left")

    #dataset_summary(active_df)

    #proportion_replies_by_gender(active_df)

    #proportion_target_replies_by_gender(active_df, label="pred_binary")

    gbv_count = active_df.groupby("mp_handle")["pred_binary"].sum().reset_index()
    appearance_count = active_df.groupby("mp_handle")["pred_contains_appearance"].sum().reset_index()

    active_mp_details = active_mp_details.merge(gbv_count, on="mp_handle", how="left")
    active_mp_details = active_mp_details.merge(appearance_count, on="mp_handle", how="left")

    #print(active_mp_details[["mp_handle", "reply_count","pred_binary", "pred_contains_appearance"]].head(10))

    active_mp_details["percentage_appearance_replies"] = active_mp_details["pred_contains_appearance"] / active_mp_details["reply_count"] * 100
    active_mp_details["percentage_gbv_replies"] = active_mp_details["pred_binary"] / active_mp_details["reply_count"] * 100


    assert active_mp_details["percentage_appearance_replies"].isna().sum() == 0, "There are MPs in active_reply_counts with no percentage of appearance related replies."
    assert active_mp_details["percentage_gbv_replies"].isna().sum() == 0, "There are MPs in active_reply_counts with no percentage of GBV related replies."


    #mannwhitney_by_group(active_mp_details, label="percentage_appearance_replies", group_by="gender")
    #mannwhitney_by_group(active_mp_details, label="percentage_gbv_replies", group_by="gender")


    #appearance_by_engagement_analysis(active_df, active_mp_details, label="percentage_appearance_replies")
    #appearance_by_engagement_analysis(active_df, active_mp_details, label="percentage_gbv_replies")

    active_df, active_mp_details = simplify_ethnicity(active_df, active_mp_details)
    #appearance_by_race_and_gender(active_df, active_mp_details, label="pred_binary")

    active_df, active_mp_details = handle_age_data(active_df, active_mp_details)

    exit()

    active_df, active_mp_details = appearance_by_age_and_gender(active_df, active_mp_details, label="percentage_appearance_replies")
    active_df, active_mp_details = appearance_by_age_and_gender(active_df, active_mp_details, label="percentage_gbv_replies")


    exit()


    gbv_analysis(active_df, active_mp_details)

    return active_df, active_mp_details

parser = argparse.ArgumentParser(description="Statistical analysis of appearance related replies to MPs on Bluesky")
parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
parser.add_argument("--platform", type=str, default="bluesky", help="Platform to analyze (default: bluesky)")
parser.add_argument("--new", action="store_true", help="Flag to indicate whether to load new data (default: False)")
args = parser.parse_args()


if __name__ == "__main__":
    prepped_df, prepped_mp_details = main(args)
    

    #prepped_df.to_csv("original_df_with_demographics.csv", index=False)


    exit()

# To Do: Sentiment, Relationship between google trends and appearance related replies









