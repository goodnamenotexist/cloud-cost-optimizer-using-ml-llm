def save_report(text):

    with open("cloud_cost_report.txt", "w", encoding="utf-8") as f:
        f.write("AI CLOUD COST OPTIMIZATION REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(text)

    print("Report saved as cloud_cost_report.txt")