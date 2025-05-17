# experiment.py  – run all datasets and log timings / outputs
import os, math, base64, json, time, requests
import pandas as pd
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
import plotly, plotly.express as px

# non‑interactive backend for Matplotlib
matplotlib.use("Agg")

API_URL = "http://localhost:6000/converse"


# ---------- helpers -------------------------------------------------
def replace_nan(x):
    if isinstance(x, dict):
        return {k: replace_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [replace_nan(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def build_metadata(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    meta = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(lambda d: d.name).to_dict(),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }
    # numeric ranges
    meta["numerical_ranges"] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        cmin, cmax = df[col].min(), df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            cmin, cmax = int(cmin), int(cmax)
        else:
            cmin, cmax = float(cmin), float(cmax)
        meta["numerical_ranges"][col] = {"min": cmin, "max": cmax}
    # categorical uniques
    meta["categorical_values"] = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        u = df[col].unique()
        meta["categorical_values"][col] = (
            u.tolist() if len(u) <= 20 else u[:20].tolist() + ["..."]
        )
    return replace_nan(meta)


def sandbox_execute(code: str, df: pd.DataFrame, out_img: str, idx: int):
    """Run code safely; save Matplotlib & Plotly pngs; return (ok, texts, secs)"""
    allowed = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "px": px,
        "plotly": plotly,
    }
    exec_globals = {"__builtins__": {}}
    # strip raw imports
    code_clean = "\n".join(
        [ln for ln in code.splitlines() if not ln.strip().startswith("import")]
    )
    lines = code_clean.strip().splitlines()
    last = lines[-1].strip() if lines else ""

    def is_expr(s: str) -> bool:
        return (
            not s.startswith(("print", "plt.", "sns.", "px.", "plotly."))
            and "=" not in s
            and not s.endswith(":")
        )

    plt.close("all")
    ok, out_txt = 1, []
    t0 = time.time()
    try:
        if len(lines) > 1 and is_expr(last):
            exec("\n".join(lines[:-1]), exec_globals, allowed)
            val = eval(last, exec_globals, allowed)
            if val is not None:
                out_txt.append(str(val))
        else:
            exec(code_clean, exec_globals, allowed)
    except Exception as e:
        ok, out_txt = 0, [f"Execution error: {e}"]
    secs = round(time.time() - t0, 4)

    # save Matplotlib figs
    for i, fnum in enumerate(plt.get_fignums(), 1):
        plt.figure(fnum).savefig(
            os.path.join(out_img, f"{idx}_plt_{i}.png"), dpi=100, bbox_inches="tight"
        )
    plt.close("all")

    # save any Plotly Figure objects
    for vname, obj in allowed.items():
        if "Figure" in str(type(obj)):
            try:
                obj.write_image(os.path.join(out_img, f"{idx}_{vname}.png"))
            except Exception as e:
                out_txt.append(f"Plotly save error ({vname}): {e}")

    return ok, out_txt, secs


# --------------------------------------------------------------------


def run_experiment(dataset_name, csv_path, commands_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jdir, idir, adir = (os.path.join(out_dir, p) for p in ("json", "images", "audio"))
    for d in (jdir, idir, adir):
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(csv_path)
    metadata = build_metadata(csv_path)
    commands = [c.strip() for c in open(commands_path, encoding="utf-8") if c.strip()]

    # running totals
    n_cmd = code_resp = code_ok = chat_ok = tts_resp = 0
    rt_sum = sandbox_sum = codegen_sum = tts_sum = 0.0
    decide_sum = chatgen_sum = 0.0
    decide_cnt = chatgen_cnt = 0
    detailed = []

    print(f"\n=== {dataset_name}: {len(commands)} commands ===")

    for idx, cmd in enumerate(commands, 1):
        payload = {
            "user_input": cmd,
            "metadata": metadata,
            "conversation_history": [],
        }
        t0 = time.time()
        try:
            r = requests.post(API_URL, json=payload)
        except Exception as e:
            print(f"[{idx}] request error: {e}")
            continue
        rt = round(time.time() - t0, 4)
        rt_sum += rt
        n_cmd += 1

        if r.status_code != 200:
            open(os.path.join(jdir, f"{idx}_error.json"), "w").write(r.text)
            continue

        data = r.json()
        open(os.path.join(jdir, f"{idx}_response.json"), "w", encoding="utf-8").write(
            json.dumps(data, indent=2)
        )

        code = data.get("code") or ""
        msg = data.get("message") or ""
        audio = data.get("audio")
        timing = data.get("timing") or {}

        # --- collect granular timing ------------------------------------
        if timing.get("decide_action_sec") is not None:
            decide_sum += timing["decide_action_sec"]
            decide_cnt += 1

        if timing.get("generate_chat_response_sec") is not None:
            chatgen_sum += timing["generate_chat_response_sec"]
            chatgen_cnt += 1

        # TTS stats
        if audio and timing.get("tts_sec") is not None:
            tts_resp += 1
            tts_sum += timing["tts_sec"]

        # code path
        sandbox_secs = 0.0
        ok_code = 0
        txt_out = []
        if code:
            code_resp += 1
            if timing.get("generate_code_sec") is not None:
                codegen_sum += timing["generate_code_sec"]
            ok_code, txt_out, sandbox_secs = sandbox_execute(code, df, idir, idx)
            sandbox_sum += sandbox_secs
            if ok_code:
                code_ok += 1
        else:
            chat_ok += 1

        # save audio
        a_path = ""
        if audio:
            try:
                a_path = os.path.join(adir, f"{idx}.wav")
                open(a_path, "wb").write(base64.b64decode(audio))
            except Exception as e:
                txt_out.append(f"audio decode error: {e}")

        detailed.append(
            {
                "idx": idx,
                "command": cmd,
                "round_trip_sec": rt,
                "server_timing": timing,
                "code_success": ok_code,
                "sandbox_sec": sandbox_secs,
                "audio_file": a_path,
                "text_out": txt_out,
            }
        )

        print(f"[{idx}] rt={rt}s code_ok={ok_code} chat={0 if code else 1}")

    # dataset summary
    summ = {
        "dataset_name": dataset_name,
        "commands_processed": n_cmd,
        "code_responses": code_resp,
        "code_success": code_ok,
        "chat_success": chat_ok,
        "tts_responses": tts_resp,
        "avg_round_trip_time_sec": round(rt_sum / n_cmd, 4) if n_cmd else 0.0,
        "avg_sandbox_time_sec": round(sandbox_sum / max(1, code_resp), 4)
        if code_resp
        else 0.0,
        "avg_decide_time_sec": round(decide_sum / max(1, decide_cnt), 4)
        if decide_cnt
        else 0.0,
        "avg_code_gen_time_sec": round(codegen_sum / max(1, code_resp), 4)
        if code_resp
        else 0.0,
        "avg_chat_resp_gen_time_sec": round(chatgen_sum / max(1, chatgen_cnt), 4)
        if chatgen_cnt
        else 0.0,
        "avg_tts_time_sec": round(tts_sum / max(1, tts_resp), 4)
        if tts_resp
        else 0.0,
    }
    json.dump(
        detailed,
        open(os.path.join(out_dir, f"{dataset_name}_results.json"), "w", encoding="utf-8"),
        indent=2,
    )
    return summ


def run_all_experiments():
    configs = [
        dict(
            dataset_name="otto_group_product",
            csv_path="exps/otto_group_product.csv",
            commands_path="exps/otto_group_product.txt",
            out_dir="out/otto",
        ),
        dict(
            dataset_name="students_performance",
            csv_path="exps/StudentsPerformance.csv",
            commands_path="exps/StudentsPerformance.txt",
            out_dir="out/students",
        ),
        dict(
            dataset_name="us_flights_2008",
            csv_path="exps/us_flights_2008.csv",
            commands_path="exps/us_flights_2008.txt",
            out_dir="out/flights",
        ),
    ]
    overall = [run_experiment(**cfg) for cfg in configs]
    os.makedirs("out", exist_ok=True)
    json.dump(
        overall, open("out/all_experiments_summary.json", "w", encoding="utf-8"), indent=2
    )

    print("\n=== Aggregate summary ===")
    for s in overall:
        print(s)


if __name__ == "__main__":
    run_all_experiments()
