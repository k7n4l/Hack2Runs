import gradio as gr
import subprocess

def run_ranker():
    cmd = [
        "python",
        "rank.py",
        "--candidates",
        "data/sample_candidates.jsonl",
        "--out",
        "output.csv"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return (
        "STDOUT:\n" + result.stdout +
        "\n\nSTDERR:\n" + result.stderr
    )

demo = gr.Interface(
    fn=run_ranker,
    inputs=[],
    outputs="text",
    title="Redrob Candidate Ranking Sandbox",
    description="Runs deterministic ranking pipeline on sample dataset"
)

demo.launch()