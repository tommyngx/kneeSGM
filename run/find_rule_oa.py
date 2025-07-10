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

def try_rule_combinations_ori(df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2, top_k=10):
    """
    Tìm top_k best rule sets cải thiện acc, f1 bằng cách refine prediction từ YOLO + model_pred.
    Trả về danh sách top_k rule sets (acc, f1, combo, report, desc).
    """
    results = []

    allowed_tokens = {"osteophyte", "osteophytemore", "osteophytebig", "narrowing", "healthy", "no detection"}

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
            desc = " AND ".join([f"(model=={r[0]} & '{r[1]}' in YOLO → {r[2]})" for r in rule_combo])
            results.append((acc, f1, rule_combo, report, desc))

    # Test combo rules: ONLY certain tokens
    combo_conditions = [
        ([3, 4], allowed_tokens, 2),
    ]
    rule_func = rule_template_factory([], combo_conditions=combo_conditions)
    acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule_func)
    desc = "If model==3 or 4 and YOLO ONLY has osteophyte/osteophytemore/osteophytebig/narrowing => 2"
    results.append((acc, f1, combo_conditions, report, desc))

    # Sort by acc, then f1, descending
    results = sorted(results, key=lambda x: (x[0], x[1]), reverse=True)
    return results[:top_k]

def try_rule_combinations(df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2, top_k=10):
    """
    Tìm top_k tổ hợp quy tắc tốt nhất để cải thiện accuracy và F1 score bằng cách kết hợp dự đoán từ YOLO và mô hình.
    Trả về: danh sách top_k (acc, f1, combo, report, desc).
    
    Parameters:
    - df: DataFrame chứa ground_truth, model_col, yolo_col
    - model_col: Tên cột chứa dự đoán của mô hình
    - yolo_col: Tên cột chứa dự đoán của YOLO
    - model_preds: Danh sách các giá trị dự đoán của mô hình
    - yolo_keywords: Danh sách các từ khóa của YOLO
    - possible_new_preds: Danh sách các giá trị dự đoán mới
    - max_rules: Số quy tắc tối đa trong một tổ hợp
    - top_k: Số tổ hợp tốt nhất cần trả về
    """
    results = []
    
    # Tập hợp từ khóa cho phép trong các quy tắc phức tạp
    allowed_tokens = {"osteophyte", "osteophytemore", "osteophytebig", "narrowing", "healthy", "no detection"}
    
    # Tạo tất cả các quy tắc tiềm năng
    all_rules = []
    for m_pred, yolo_kw, new_pred in itertools.product(model_preds, yolo_keywords, possible_new_preds):
        if new_pred > m_pred:  # Chỉ giữ các quy tắc nâng cấp nhãn
            all_rules.append((m_pred, yolo_kw, new_pred))
    
    # Thử nghiệm các tổ hợp từ 1 đến max_rules quy tắc
    for k in range(1, max_rules + 1):
        for rule_combo in itertools.combinations(all_rules, k):
            rule_func = rule_template_factory(list(rule_combo))
            acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule_func)
            desc = " AND ".join([f"(model=={r[0]} & '{r[1]}' in YOLO → {r[2]})" for r in rule_combo])
            results.append((acc, f1, rule_combo, report, desc))
    
    # Thêm quy tắc phức tạp: Chỉ chấp nhận một số từ khóa nhất định
    combo_conditions = [
        ([3, 4], allowed_tokens, 2),  # Nếu model=3 hoặc 4 và YOLO chỉ chứa các từ khóa cho phép, đổi thành 2
    ]
    rule_func = rule_template_factory([], combo_conditions=combo_conditions)
    acc, f1, report, _ = evaluate_rule(df, model_col, yolo_col, rule_func)
    desc = "If model==3 or 4 and YOLO ONLY has osteophyte/osteophytemore/osteophytebig/narrowing => 2"
    results.append((acc, f1, combo_conditions, report, desc))
    
    # Sắp xếp theo accuracy và F1 score (giảm dần)
    results = sorted(results, key=lambda x: (x[0], x[1]), reverse=True)
    return results[:top_k]

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

    # Baseline: no rule, just model prediction
    baseline_acc = accuracy_score(df["ground_truth"], df[model_col])
    baseline_f1 = f1_score(df["ground_truth"], df[model_col], average="macro")
    print(f"\nBaseline (no rule): Accuracy: {baseline_acc:.4f}, F1: {baseline_f1:.4f}")

    print("\nSearching for top rule combinations...")
    top_rules = try_rule_combinations(
        df, model_col, yolo_col, model_preds, yolo_keywords, possible_new_preds, max_rules=2, top_k=10
    )
    for i, (acc, f1, combo, report, desc) in enumerate(top_rules, 1):
        improved = ""
        if acc > baseline_acc or (acc == baseline_acc and f1 > baseline_f1):
            improved = " (IMPROVED)"
        print(f"\nTop {i}:{improved}")
        print(desc)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        #print("Classification report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best rule-based OA refinement from CSV.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--model_col', type=str, default='resnet50', help="Model prediction column name")
    parser.add_argument('--yolo_col', type=str, default='YOLO_prediction', help="YOLO prediction column name (default: YOLO_prediction)")
    args = parser.parse_args()
    main(args.csv_path, args.model_col, args.yolo_col)
