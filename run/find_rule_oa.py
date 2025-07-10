import pandas as pd
import argparse
import itertools
from sklearn.metrics import accuracy_score, f1_score, classification_report

def rule_template_factory(rules, combo_conditions=[]):
    """
    Trả về hàm rule(model_pred, yolo_text) áp dụng danh sách rules và combo_rules.
    """
    def rule(model_pred, yolo_text):
        text = yolo_text.lower() if isinstance(yolo_text, str) else ""
        tokens = {token.strip() for token in text.replace(";", ",").split(",") if token.strip()}

        # Apply single rules
        for m_pred, yolo_kw, new_pred in rules:
            if model_pred == m_pred and yolo_kw in text:
                return new_pred

        # Apply combo conditions (model in set, YOLO tokens subset allowed_tokens)
        for m_pred_set, allowed_tokens, new_pred in combo_conditions:
            if isinstance(m_pred_set, int):
                m_pred_set = [m_pred_set]
            if model_pred in m_pred_set and tokens and tokens.issubset(allowed_tokens):
                return new_pred

        return model_pred
    return rule

def evaluate_rule(df, model_col, yolo_col, rule_func):
    preds = []
    for _, row in df.iterrows():
        preds.append(rule_func(row[model_col], row[yolo_col]))
    acc = accuracy_score(df["ground_truth"], preds)
    f1 = f1_score(df["ground_truth"], preds, average="macro")
    report = classification_report(df["ground_truth"], preds)
    return acc, f1, report, preds

def try_rule_combinations(df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2):
    """
    Tìm best rule set cải thiện acc, f1 bằng cách refine prediction từ YOLO + model_pred.
    """
    best_acc = 0
    best_f1 = 0
    best_combo = None
    best_report = ""
    best_desc = None

    allowed_tokens = {"osteophyte", "osteophytemore", "osteophytebig", "narrowing"}

    # Tạo rule tuples (m_pred, yolo_kw, new_pred) hợp lý
    all_rules = []
    for m_pred, yolo_kw, new_pred in itertools.product(model_preds, yolo_keywords, possible_new_preds):
        if new_pred > m_pred:  # chỉ test khi nâng nhãn
            all_rules.append((m_pred, yolo_kw, new_pred))

    # Test combinations of 1 and 2 rules
    for k in range(1, max_rules + 1):
        for rule_combo in itertools.combinations(all_rules, k):
            rule_func = rule_template_factory(list(rule_combo))
            acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule_func)
            if acc > best_acc or (acc == best_acc and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_combo = rule_combo
                best_report = report
                best_desc = " AND ".join([f"(model=={r[0]} & '{r[1]}' in YOLO → {r[2]})" for r in rule_combo])

    # Test combo rules: ONLY certain tokens
    combo_conditions = [
        ([3, 4], allowed_tokens, 2),
    ]
    rule_func = rule_template_factory([], combo_conditions=combo_conditions)
    acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule_func)
    if acc > best_acc or (acc == best_acc and f1 > best_f1):
        best_acc = acc
        best_f1 = f1
        best_combo = combo_conditions
        best_report = report
        best_desc = "If model==3 or 4 and YOLO ONLY has osteophyte/osteophytemore/osteophytebig/narrowing => 2"

    return best_acc, best_f1, best_combo, best_report, best_desc

def main(csv_path, model_col, yolo_col):
    df = pd.read_csv(csv_path)
    model_preds = sorted(df[model_col].unique())
    yolo_keywords = set()
    for y in df[yolo_col].dropna():
        for token in str(y).lower().replace(";",",").split(","):
            token = token.strip()
            if token:
                yolo_keywords.add(token)
    yolo_keywords = sorted(list(yolo_keywords))
    possible_new_preds = sorted(df[model_col].unique())

    print("\nSearching for best rule combinations...")
    combo_acc, combo_f1, combo_rules, combo_report, combo_desc = try_rule_combinations(
        df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2
    )
    print("Best rule or combo found:")
    print(combo_desc)
    print(f"Accuracy: {combo_acc:.4f}, F1: {combo_f1:.4f}")
    print("Classification report:\n", combo_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best rule-based OA refinement from CSV.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--model_col', type=str, default='resnet50', help="Model prediction column name")
    parser.add_argument('--yolo_col', type=str, default='YOLO_prediction', help="YOLO prediction column name")
    args = parser.parse_args()
