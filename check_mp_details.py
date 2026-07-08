import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('FINAL-mp-details.csv')

print(raw_data["minority_status"].value_counts())

raw_data["has_bluesky"] = raw_data["bluesky_handle"].apply(lambda x: pd.notna(x) and x != "")
print(raw_data["has_bluesky"].value_counts())

raw_data["has_twitter"] = raw_data["twitter_handle"].apply(lambda x: pd.notna(x) and x != "")
print(raw_data["has_twitter"].value_counts())

def has_account_by_gender(mp_details):
    cross_tab = pd.crosstab(mp_details["has_bluesky"], mp_details["gender"])
    #chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
    #print(f"Chi-squared test results for Bluesky presence by gender: chi2={chi2}, p-value={p}, dof={dof}")

    cross_tab = pd.crosstab(mp_details["gender"], mp_details["has_bluesky"],  normalize='index')

    plt.figure(figsize=(6, 4))
    cross_tab.plot(kind="bar", stacked=True, color=["red", "blue"], alpha=0.4)
    plt.xlabel("Has Bluesky Account")
    plt.ylabel("Number of MPs")
    plt.title("Bluesky Account Presence by Gender")
    plt.legend(title="Has Bluesky Account", labels=["No", "Yes"])
    plt.xticks(rotation=0)
    plt.show()
    plt.savefig("bluesky_presence_by_gender.png")

#has_account_by_gender(raw_data)

def has_account_by_age(mp_details, platform="bluesky"):
    # Histogram of age distribution of MPs, excluding those with unknown birth year
    plt.figure(figsize=(10, 5))
    mp_details_filtered = mp_details[~mp_details["wiki_birth_year"].isna()]
    mp_details_filtered["age"] = 2026 - mp_details_filtered["wiki_birth_year"].astype(int)
    #plt.hist(mp_details_filtered["wiki_birth_year"], bins=range(1939, 2005, 5), edgecolor="black", alpha=0.2, label="MPs")
    plt.hist(mp_details_filtered[mp_details_filtered[f"has_{platform}"]]["age"], bins=range(20, 85, 5), color="blue", alpha=0.4, label=f"MPs with {platform.capitalize()}")
    plt.hist(mp_details_filtered[~mp_details_filtered[f"has_{platform}"]]["age"], bins=range(20, 85, 5), color="red", alpha=0.4, label=f"MPs without {platform.capitalize()}")

    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title(f"Age Distribution of MPs, with and without {platform.capitalize()} Accounts")
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig(f"age_distribution_mps_{platform}.png")

has_account_by_age(raw_data, platform="twitter")