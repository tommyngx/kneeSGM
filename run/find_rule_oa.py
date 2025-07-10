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

def try_rule_combinations(df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2):
    """
    Try combinations of up to max_rules rules and return the best combination.
    """
    best_acc = 0
    best_f1 = 0
    best_combo = None
    best_report = ""
    best_desc = None

    # Generate all possible rule tuples (m_pred, yolo_kw, new_pred)
    all_rules = []
    for m_pred, yolo_kw, new_pred in itertools.product(model_preds, yolo_keywords, possible_new_preds):
        if new_pred != m_pred:
            all_rules.append((m_pred, yolo_kw, new_pred))

    # Try all combinations of 2 rules (can increase max_rules for more, but will be slow)
    for rule_combo in itertools.combinations(all_rules, max_rules):
        rule = rule_template_factory(list(rule_combo))
        acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule)
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_combo = rule_combo
            best_report = report
            best_desc = " AND ".join([f"(model=={r[0]} & '{r[1]}' in YOLO → {r[2]})" for r in rule_combo])

    return best_acc, best_f1, best_combo, best_report, best_desc

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

    # Single rule search (existing code)
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

    # Try combinations of 2 rules for better performance
    print("\nSearching for best combination of 2 rules...")
    combo_acc, combo_f1, combo_rules, combo_report, combo_desc = try_rule_combinations(
        df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2
    )
    print("Best 2-rule combination found:")
    print(combo_desc)
    print(f"Accuracy: {combo_acc:.4f}, F1: {combo_f1:.4f}")
    print("Classification report:\n", combo_report)

    # You can increase max_rules in try_rule_combinations for more complex rules (may be slow).

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best rule-based OA refinement from CSV.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--model_col', type=str, default='resnet50', help="Model prediction column name")
    parser.add_argument('--yolo_col', type=str, default='YOLO_prediction', help="YOLO prediction column name")
    args = parser.parse_args()
    main(args.csv_path, args.model_col, args.yolo_col)
