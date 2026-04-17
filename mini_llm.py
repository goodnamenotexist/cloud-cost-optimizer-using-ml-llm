import random

# ── Load training data ─────────────────────────────────────────
with open("llm_training_data.txt") as f:
    text = f.read()

words = text.split()

# Build Markov model
model = {}

for i in range(len(words) - 1):
    key = words[i]
    next_word = words[i + 1]

    if key not in model:
        model[key] = []

    model[key].append(next_word)


# ── Generate text ──────────────────────────────────────────────
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


# ── Cost calculation ───────────────────────────────────────────
def compute_savings(cost):
    saving_pct = random.randint(10, 40)
    savings = cost * (saving_pct / 100)

    return {
        "saving_pct": saving_pct,
        "savings": round(savings, 4),
        "optimized_cost": round(cost - savings, 4)
    }


# ── Main function (LLM) ────────────────────────────────────────
def generate_suggestion(prediction, data):

    generated_text = generate_text()

    cost = data.get("cost", 0)
    savings = compute_savings(cost)

    if prediction == 1:

        return f"""
==================================================
   AI CLOUD COST OPTIMIZATION REPORT
==================================================

ML Prediction: Inefficient Resource

AI Insight:
{generated_text}

Recommendations:
• Optimize resource allocation
• Downsize underutilized instances
• Improve cloud efficiency

Estimated Cost:
Current Cost   : ${cost}
Optimized Cost : ${savings['optimized_cost']}
Savings        : ${savings['savings']} ({savings['saving_pct']}%)

==================================================
"""

    else:

        return f"""
==================================================
   AI CLOUD COST OPTIMIZATION REPORT
==================================================

ML Prediction: Efficient Resource

AI Insight:
System operating efficiently with balanced resource utilization.

Recommendation:
No immediate action required. Continue monitoring.

==================================================
"""