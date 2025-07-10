import pandas as pd
import argparse
import itertools
from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_rule(df, model_col, yolo_col, rule):
    """
    Apply a rule function to each row and return accuracy, f1, and report.
    """
    preds = []
    for idx, row in df.iterrows():
        preds.append(rule(row[model_col], row[yolo_col]))
    acc = accuracy_score(df['ground_truth'], preds)
    f1 = f1_score(df['ground_truth'], preds, average='weighted')
    report = classification_report(df['ground_truth'], preds, zero_division=0)
    return acc, f1, report, preds

def rule_template_factory(conditions):
    """
    Return a rule function based on a list of (model_pred, yolo_keyword, new_pred) tuples.
    """
    def rule(model_pred, yolo_text):
        text = str(yolo_text).lower()
        for cond in conditions:
            m_pred, yolo_kw, new_pred = cond
            if (model_pred == m_pred) and (yolo_kw in text):
                return new_pred
        return model_pred
    return rule

def main(csv_path, model_col, yolo_col):
    df = pd.read_csv(csv_path)
    # Define possible rules to try (expand as needed)
    model_preds = sorted(df[model_col].unique())
    yolo_keywords = set()
    for y in df[yolo_col].dropna():
        for token in str(y).lower().replace(";",",").split(","):
            token = token.strip()
            if token:
                yolo_keywords.add(token)
    yolo_keywords = sorted(list(yolo_keywords))
    possible_new_preds = sorted(df[model_col].unique())

    # Generate all possible single rules (for demonstration)
    best_acc = 0
    best_rule = None
    best_desc = None
    best_f1 = 0
    best_report = ""
    for m_pred, yolo_kw, new_pred in itertools.product(model_preds, yolo_keywords, possible_new_preds):
        if new_pred == m_pred:
            continue
        rule = rule_template_factory([(m_pred, yolo_kw, new_pred)])
        acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule)
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_rule = (m_pred, yolo_kw, new_pred)
            best_report = report
            best_desc = f"If model=={m_pred} and '{yolo_kw}' in YOLO, predict {new_pred}"

    print("Best single rule found:")
    print(best_desc)
    print(f"Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")
    print("Classification report:\n", best_report)

    # You can expand this to try combinations of rules, or more complex logic.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best rule-based OA refinement from CSV.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--model_col', type=str, default='resnet50', help="Model prediction column name")
    parser.add_argument('--yolo_col', type=str, default='YOLO_prediction', help="YOLO prediction column name")
    args = parser.parse_args()
    main(args.csv_path, args.model_col, args.yolo_col)
