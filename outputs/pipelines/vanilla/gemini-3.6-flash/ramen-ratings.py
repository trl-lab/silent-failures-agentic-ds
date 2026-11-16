sub = pd.DataFrame({"id": np.arange(len(test_df)), "Stars": final_stars})
sub.to_csv("predictions.csv", index=False)