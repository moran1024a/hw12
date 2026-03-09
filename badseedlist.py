import csv
import os

# 改这里：原始 csv 文件相对路径
input_csv = "./badcase_logs/badcase_seed_pool.csv"

# 输出文件：和原文件放在同一目录
output_csv = os.path.join(os.path.dirname(input_csv), "badseedlist.csv")

result_rows = []
total_rows = 0
case0_count = 0
case1_count = 0

with open(input_csv, "r", encoding="utf-8-sig", newline="") as infile:
    reader = csv.DictReader(infile)

    # 清理表头空格
    reader.fieldnames = [name.strip() for name in reader.fieldnames]

    for row in reader:
        total_rows += 1

        # 清理每个字段的 key/value
        clean_row = {}
        for k, v in row.items():
            clean_key = k.strip() if k else k
            clean_value = v.strip() if isinstance(v, str) else v
            clean_row[clean_key] = clean_value

        try:
            steps = float(clean_row["steps"])
            seed = clean_row["seed"]
        except (KeyError, ValueError, TypeError):
            print(f"跳过第 {total_rows} 行，数据异常: {clean_row}")
            continue

        if 50 < steps < 500:
            result_rows.append({
                "case": 0,
                "badseed": seed
            })
            case0_count += 1

        elif steps > 900:
            result_rows.append({
                "case": 1,
                "badseed": seed
            })
            case1_count += 1

with open(output_csv, "w", encoding="utf-8-sig", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["case", "badseed"])
    writer.writeheader()
    writer.writerows(result_rows)

print(f"总行数: {total_rows}")
print(f"case=0 数量: {case0_count}")
print(f"case=1 数量: {case1_count}")
print(f"输出文件: {output_csv}")
