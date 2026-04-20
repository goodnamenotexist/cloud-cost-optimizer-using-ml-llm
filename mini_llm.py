import random
import os

file_path = os.path.join(os.path.dirname(__file__), "llm_training_data.txt")

with open(file_path) as f:
    text = f.read()

words = text.split()

model = {}
for i in range(len(words) - 1):
    key = words[i]
    next_word = words[i + 1]
    if key not in model:
        model[key] = []
    model[key].append(next_word)

def generate_text(length=30):
    word = random.choice(list(model.keys()))
    result = [word]
    for _ in range(length):
        if word in model:
            word = random.choice(model[word])
            result.append(word)
        else:
            break
    return " ".join(result)


RESOURCE_THRESHOLDS = {
    "cpu":        {"high": 85, "low": 15, "unit": "%"},
    "memory":     {"high": 88, "low": 18, "unit": "%"},
    "disk_io":    {"high": 80, "low": 10, "unit": "%"},
    "network":    {"high": 75, "low": 8,  "unit": "%"},
    "gpu":        {"high": 80, "low": 15, "unit": "%"},
    "idle_time":  {"high": 50, "low": 5,  "unit": "%"},
}

def analyze_resources(resource_metrics: dict) -> dict:
 
    issues = []
    recommendations = []
    score = 100
    resource_status = {}

    cpu = resource_metrics.get("cpu", 0)
    memory = resource_metrics.get("memory", 0)
    disk_io = resource_metrics.get("disk_io", 0)
    network = resource_metrics.get("network", 0)
    gpu = resource_metrics.get("gpu", 0)
    idle = resource_metrics.get("idle_time", 0)
    instance = resource_metrics.get("instance_type", "general")
    cost = resource_metrics.get("cost", 0)

    if cpu > 85:
        issues.append("CPU overloaded")
        recommendations.append({
            "resource": "CPU",
            "action": "Upgrade to a higher CPU tier or add horizontal scaling",
            "saving": 0,
            "priority": "High"
        })
        score -= 20
        resource_status["cpu"] = "Overloaded"
    elif cpu < 15:
        issues.append("CPU heavily underutilized")
        recommendations.append({
            "resource": "CPU",
            "action": "Downsize to a smaller instance type (e.g. t3.medium → t3.small)",
            "saving": round(cost * 0.22, 2),
            "priority": "High"
        })
        score -= 18
        resource_status["cpu"] = "Underutilized"
    else:
        resource_status["cpu"] = "Healthy"

    if memory > 88:
        issues.append("Memory critically high — risk of OOM")
        recommendations.append({
            "resource": "Memory",
            "action": "Migrate to memory-optimized instance (e.g. r5.xlarge)",
            "saving": round(-cost * 0.08, 2),  # negative = extra cost needed
            "priority": "Critical"
        })
        score -= 18
        resource_status["memory"] = "Critical"
    elif memory < 18:
        issues.append("Memory underutilized")
        recommendations.append({
            "resource": "Memory",
            "action": "Reduce memory allocation or switch to smaller instance",
            "saving": round(cost * 0.15, 2),
            "priority": "Medium"
        })
        score -= 12
        resource_status["memory"] = "Underutilized"
    else:
        resource_status["memory"] = "Healthy"

    if idle > 50:
        issues.append("Very high idle time — resource is mostly unused")
        recommendations.append({
            "resource": "Idle Time",
            "action": "Enable auto-scaling and schedule shutdowns during off-peak hours",
            "saving": round(cost * 0.30, 2),
            "priority": "Critical"
        })
        score -= 25
        resource_status["idle_time"] = "Wasteful"
    elif idle > 30:
        issues.append("Moderate idle time detected")
        recommendations.append({
            "resource": "Idle Time",
            "action": "Configure scheduled stop/start policies for non-production hours",
            "saving": round(cost * 0.12, 2),
            "priority": "Medium"
        })
        score -= 10
        resource_status["idle_time"] = "Moderate"
    else:
        resource_status["idle_time"] = "Good"

    if disk_io < 10:
        issues.append("Disk I/O very low — storage tier may be over-provisioned")
        recommendations.append({
            "resource": "Disk I/O",
            "action": "Switch from io1 to gp3 storage — same performance, lower cost",
            "saving": round(cost * 0.09, 2),
            "priority": "Low"
        })
        score -= 8
        resource_status["disk_io"] = "Over-provisioned"
    elif disk_io > 80:
        issues.append("Disk I/O bottleneck detected")
        recommendations.append({
            "resource": "Disk I/O",
            "action": "Upgrade to provisioned IOPS (io2) or add caching layer (ElastiCache/Redis)",
            "saving": round(-cost * 0.05, 2),
            "priority": "High"
        })
        score -= 12
        resource_status["disk_io"] = "Bottleneck"
    else:
        resource_status["disk_io"] = "Healthy"

    if network < 8:
        issues.append("Network bandwidth barely used")
        recommendations.append({
            "resource": "Network",
            "action": "Review data transfer costs — consider VPC endpoints to reduce egress fees",
            "saving": round(cost * 0.05, 2),
            "priority": "Low"
        })
        score -= 5
        resource_status["network"] = "Underused"
    elif network > 75:
        issues.append("Network nearing saturation")
        recommendations.append({
            "resource": "Network",
            "action": "Enable enhanced networking or move to network-optimized instance",
            "saving": 0,
            "priority": "Medium"
        })
        score -= 10
        resource_status["network"] = "Saturated"
    else:
        resource_status["network"] = "Healthy"

    if instance == "gpu" and gpu < 20:
        issues.append("GPU instance with very low GPU utilization — very expensive waste")
        recommendations.append({
            "resource": "GPU",
            "action": "Migrate workload to a CPU instance or use spot GPU instances only when needed",
            "saving": round(cost * 0.45, 2),
            "priority": "Critical"
        })
        score -= 22
        resource_status["gpu"] = "Wasted"
    elif gpu > 80:
        resource_status["gpu"] = "High load"
    else:
        resource_status["gpu"] = "Normal"

    if not issues:
        recommendations.append({
            "resource": "Billing",
            "action": "Switch to 1-year reserved instance pricing for ~30% discount",
            "saving": round(cost * 0.30, 2),
            "priority": "Medium"
        })

    score = max(5, min(100, score))
    total_saving = sum(r["saving"] for r in recommendations if r["saving"] > 0)

    return {
        "score": score,
        "status": "Efficient" if score >= 75 else "Needs Review" if score >= 45 else "Inefficient",
        "issues": issues,
        "resource_status": resource_status,
        "recommendations": recommendations,
        "total_saving": round(total_saving, 2),
        "optimized_cost": round(max(0, cost - total_saving), 2)
    }

def compute_savings(cost):
    saving_pct = random.randint(10, 40)
    savings = cost * (saving_pct / 100)
    return {
        "saving_pct": saving_pct,
        "savings": round(savings, 4),
        "optimized_cost": round(cost - savings, 4)
    }

def format_report(analysis: dict, data: dict) -> str:
    cost = data.get("cost", 0)
    score = analysis["score"]
    status = analysis["status"]

    filled = int(score / 5)
    bar = "█" * filled + "░" * (20 - filled)

    lines = [
        "=" * 54,
        "   AI CLOUD COST OPTIMIZATION REPORT",
        "=" * 54,
        "",
        f"  Efficiency Score : [{bar}] {score}/100",
        f"  Status           : {status}",
        f"  Current Cost     : ${cost}",
        f"  Optimized Cost   : ${analysis['optimized_cost']}",
        f"  Potential Savings: ${analysis['total_saving']}/mo",
        "",
        "─" * 54,
        "  RESOURCE HEALTH",
        "─" * 54,
    ]

    for resource, status_text in analysis["resource_status"].items():
        label = resource.replace("_", " ").title().ljust(14)
        icon = "✅" if status_text in ("Healthy", "Good", "Normal") else "❌"
        lines.append(f"  {icon}  {label}  {status_text}")

    if analysis["issues"]:
        lines += [
            "",
            "─" * 54,
            "  ISSUES DETECTED",
            "─" * 54,
        ]
        for issue in analysis["issues"]:
            lines.append(f"  • {issue}")

    lines += [
        "",
        "─" * 54,
        "  RECOMMENDATIONS",
        "─" * 54,
    ]

    for i, rec in enumerate(analysis["recommendations"], 1):
        saving_str = f"  → Save ${rec['saving']}/mo" if rec["saving"] > 0 else \
                     f"  → Extra cost: ${abs(rec['saving'])}/mo" if rec["saving"] < 0 else ""
        lines.append(f"  {i}. [{rec['priority']}] {rec['resource']}")
        lines.append(f"     {rec['action']}")
        if saving_str:
            lines.append(saving_str)
        lines.append("")

    lines += [
        "─" * 54,
        "  AI INSIGHT",
        "─" * 54,
        f"  {generate_text(40)}",
        "",
        "=" * 54,
    ]

    return "\n".join(lines)


def generate_suggestion(prediction: int, data: dict) -> str:

    analysis = analyze_resources(data)

    if prediction == 0 and analysis["score"] < 75:
        analysis["score"] = max(analysis["score"], 75)
        analysis["status"] = "Efficient"

    return format_report(analysis, data)


def interactive_mode():
    print("\n" + "=" * 54)
    print("   CLOUD COST OPTIMIZER — Interactive Mode")
    print("=" * 54)

    try:
        cost        = float(input("\n  Monthly cost ($): ") or 1200)
        cpu         = float(input("  CPU usage (%): ") or 72)
        memory      = float(input("  Memory usage (%): ") or 85)
        disk_io     = float(input("  Disk I/O (%): ") or 34)
        network     = float(input("  Network usage (%): ") or 15)
        gpu         = float(input("  GPU usage (%): ") or 20)
        idle_time   = float(input("  Idle time (%): ") or 40)
        prediction  = int(input("  ML Prediction (1=inefficient, 0=efficient): ") or 1)

        print("\n  Instance types: compute | memory | general | gpu | storage")
        instance = input("  Instance type [general]: ").strip() or "general"

    except (ValueError, KeyboardInterrupt):
        print("\n  Invalid input. Using defaults.")
        cost, cpu, memory, disk_io = 1200, 72, 85, 34
        network, gpu, idle_time    = 15, 20, 40
        prediction, instance       = 1, "general"

    data = {
        "cost": cost,
        "cpu": cpu,
        "memory": memory,
        "disk_io": disk_io,
        "network": network,
        "gpu": gpu,
        "idle_time": idle_time,
        "instance_type": instance
    }

    report = generate_suggestion(prediction, data)
    print("\n" + report)

if __name__ == "__main__":
    interactive_mode()