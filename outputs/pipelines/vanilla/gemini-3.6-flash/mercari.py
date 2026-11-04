submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'price': final_pred
})
submission.to_csv('predictions.csv', index=False)