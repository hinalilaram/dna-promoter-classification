from tqdm import tqdm
from utils import DataUtils, evaluate_model, results_table
from model import Model

def main():
    # Load dataset
    dataset_name = "dnagpt/dna_core_promoter"
    dataset = DataUtils(dataset_name)
    train_ds, val_ds, test_ds = dataset["train"], dataset["validation"], dataset["test"]

    # Extract sequences and labels
    X_train_texts = train_ds["sequence"]
    y_train = train_ds["label"]
    X_val_texts = val_ds["sequence"]
    y_val = val_ds["label"]
    X_test_texts = test_ds["sequence"]
    y_test = test_ds["label"]

    # Initialize model (automatically loads model/tokenizer)
    
    model = Model("zhangtaolab/dnabert2-promoter")

    #print(train_ds)
    hf_model = Model().load()
        # Tokenize datasets
    train_ds = hf_model.preprocess(train_ds)
    val_ds = hf_model.preprocess(val_ds)
    test_ds = hf_model.preprocess(test_ds)

    #print(val_ds)
    # Get Trainer
    trainer = hf_model.get_trainer(train_ds, val_ds, epochs=20)

    # Train
    trainer.train()

    # Evaluate
    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    # Extract embeddings from the model instance
    # X_train = model.extract_embeddings(X_train_texts)
    # X_val = model.extract_embeddings(X_val_texts)
    # X_test = model.extract_embeddings(X_test_texts)

    # print("Embeddings extracted:")
    # print(f"Train embeddings shape: {X_train.shape}")
    # print(f"Validation embeddings shape: {X_val.shape}")
    # print(f"Test embeddings shape: {X_test.shape}")
    # # Run ML classifiers on embeddings
    # df_results = run_experiments(X_train, X_val, y_train, y_val)

    # Evaluate on test set using the best classifier
    # best_clf_name = df_results['accuracy'].idxmax()
    # best_clf = Model.get_classifiers()[best_clf_name]
    # best_clf.fit(X_train, y_train)
    # test_preds = best_clf.predict(X_test)
    # test_metrics = evaluate_model(y_test, test_preds)
    # print("Test metrics for best classifier:")
    # print(test_metrics)

def run_experiments(X_train, X_val, y_train, y_val):

    
    classifiers = Model.get_classifiers()
    results = {}

    for name, clf in tqdm(classifiers.items(), desc="Training classifiers"):
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        metrics = evaluate_model(y_val, preds)
        results[name] = metrics

    df_results = results_table(results)
    print(df_results)
    return df_results

if __name__ == "__main__":
    main()
