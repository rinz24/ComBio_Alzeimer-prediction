import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import load_subject_raw
from preprocessing import preprocess
from feature_extract import extract_features
from dataset import build_dataset
from ml_models import run_optuna_training
from plots import show_single_model_graphs

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline




# ============================================================
# MAIN GUI CLASS
# ============================================================
class AlzheimerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Alzheimer EEG Detector — Sidebar Edition")
        self.geometry("1100x650")

        self.history = {}
        self.raw = None

        # Sidebar
        self.sidebar = tk.Frame(self, bg="#222", width=200)
        self.sidebar.pack(side="left", fill="y")

        # Main content area
        self.main = tk.Frame(self, bg="white")
        self.main.pack(side="right", fill="both", expand=True)

        # Sidebar buttons
        buttons = [
            ("Home", self.show_home),
            ("Test Date", self.show_date),
            ("Alzheimer Probability", self.show_probability),
            ("EEG Viewer", self.show_eeg),
            ("Compare Patients", self.show_compare),
            ("Exit", self.quit)
        ]

        for text, cmd in buttons:
            tk.Button(
                self.sidebar, text=text, font=("Arial", 14),
                bg="#333", fg="white", relief="flat",
                command=cmd, height=2
            ).pack(fill="x", pady=2)

        self.show_home()




    # ============================================================
    def clear_main(self):
        for w in self.main.winfo_children():
            w.destroy()




    # ============================================================
    # HOME SCREEN
    # ============================================================
    def show_home(self):
        self.clear_main()
        tk.Label(self.main, text="Select Patient", font=("Arial", 26)).pack(pady=20)

        tk.Label(self.main, text="Paciente:").pack()
        self.pEntry = tk.Entry(self.main, font=("Arial", 14))
        self.pEntry.insert(0, self.history.get("paciente", "Paciente1"))
        self.pEntry.pack()

        tk.Label(self.main, text="Group:").pack()
        self.group = tk.StringVar(value=self.history.get("group", "AD"))
        ttk.Combobox(self.main, textvariable=self.group, values=["AD", "Healthy"]).pack()

        tk.Label(self.main, text="State:").pack()
        self.state = tk.StringVar(value=self.history.get("state", "Eyes_open"))
        ttk.Combobox(self.main, textvariable=self.state, values=["Eyes_open", "Eyes_closed"]).pack()

        tk.Button(
            self.main, text="Load Patient", font=("Arial", 14),
            command=self.load_patient
        ).pack(pady=20)


    def load_patient(self):
        paciente = self.pEntry.get()
        group = self.group.get()
        state = self.state.get()

        raw = load_subject_raw(group, state, paciente)

        if raw is None:
            messagebox.showerror("Error", "Cannot load EEG for this patient.")
            return

        self.raw = raw
        self.history.update({"paciente": paciente, "group": group, "state": state})

        messagebox.showinfo("Success", "Patient loaded successfully.")




    # ============================================================
    # DATE SCREEN
    # ============================================================
    def show_date(self):
        self.clear_main()
        tk.Label(self.main, text="Test Date", font=("Arial", 26)).pack(pady=20)

        tk.Label(self.main, text="Select Date:").pack()
        self.dateEntry = tk.Entry(self.main, font=("Arial", 14))
        self.dateEntry.insert(0, self.history.get("date", datetime.now().strftime("%Y-%m-%d")))
        self.dateEntry.pack()

        tk.Button(
            self.main, text="Save", font=("Arial", 14),
            command=self.save_date
        ).pack(pady=15)


    def save_date(self):
        self.history["date"] = self.dateEntry.get()
        messagebox.showinfo("Saved", "Date saved successfully.")




    # ============================================================
    # ALZHEIMER PROBABILITY SCREEN
    # ============================================================
    def show_probability(self):
        self.clear_main()
        tk.Label(self.main, text="Alzheimer Probability", font=("Arial", 26)).pack(pady=20)

        if self.raw is None:
            tk.Label(self.main, text="Load a patient first.", font=("Arial", 16)).pack()
            return

        Xp = preprocess(self.raw)
        prob = self.compute_slowing(Xp)

        tk.Label(
            self.main, text=f"{prob:.2f}% Alzheimer-like EEG",
            font=("Arial", 22)
        ).pack(pady=20)

        tk.Button(
            self.main, text="Run ML Models", font=("Arial", 14),
            command=self.run_ml
        ).pack(pady=15)


    def compute_slowing(self, Xp):
        from scipy.signal import welch
        f, P = welch(Xp.mean(axis=0), 256)

        tot = np.trapz(P[(f >= 0.5) & (f <= 45)], f[(f >= 0.5) & (f <= 45)]) + 1e-9
        d = np.trapz(P[(f >= 0.5) & (f <= 4)], f[(f >= 0.5) & (f <= 4)]) / tot
        t = np.trapz(P[(f >= 4) & (f <= 8)], f[(f >= 4) & (f <= 8)]) / tot
        a = np.trapz(P[(f >= 8) & (f <= 13)], f[(f >= 8) & (f <= 13)]) / tot
        b = np.trapz(P[(f >= 13) & (f <= 30)], f[(f >= 13) & (f <= 30)]) / tot

        si = (d + t) / (a + b + 1e-12)
        return 100 * (1 / (1 + np.exp(-(si - 1) * 2)))




    # ============================================================
    # MACHINE LEARNING (OPTUNA)
    # ============================================================
    def run_ml(self):
        if self.raw is None:
            messagebox.showerror("Error", "No patient loaded.")
            return

        (
            best_model_name,
            acc, f1, auc,
            cm,
            best_params,
            patient_prob
        ) = run_optuna_training(self.history.get("state", "Eyes_open"), self.raw)

        # Fix: best_params might be array instead of dict
        if isinstance(best_params, dict):
            formatted_params = "\n".join([f"{k}: {v}" for k, v in best_params.items()])
        elif isinstance(best_params, (list, tuple, np.ndarray)):
            formatted_params = "\n".join([f"{i}: {p}" for i, p in enumerate(best_params)])
        else:
            formatted_params = str(best_params)

        messagebox.showinfo(
            "Prediction",
            f"Best Model: {best_model_name}\n"
            f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\nF1: {f1:.3f}\n\n"
            f"Alzheimer Probability: {patient_prob:.2f}%\n\n"
            f"Best Params:\n{formatted_params}"
        )

        show_single_model_graphs(acc, f1, auc, cm, best_model_name)




    # ============================================================
    # EEG VIEWER (BUTTONS FIXED)
    # ============================================================
    def show_eeg(self):
        self.clear_main()
        tk.Label(self.main, text="EEG Viewer", font=("Arial", 26)).pack(pady=20)

        if self.raw is None:
            tk.Label(self.main, text="Load a patient first.", font=("Arial", 16)).pack()
            return

        # Buttons are now ALWAYS visible
        tk.Button(
            self.main, text="Show Raw EEG", font=("Arial", 14),
            command=self.plot_raw
        ).pack(pady=10)

        tk.Button(
            self.main, text="Show Filtered EEG", font=("Arial", 14),
            command=self.plot_filtered
        ).pack(pady=10)


    def plot_raw(self):
        if self.raw is None:
            return
        plt.figure(figsize=(10,6))
        for i, ch in enumerate(self.raw):
            plt.plot(ch + i*5)
        plt.title("Raw EEG")
        plt.show()


    def plot_filtered(self):
        if self.raw is None:
            return
        Xp = preprocess(self.raw)
        plt.figure(figsize=(10,6))
        for i, ch in enumerate(Xp):
            plt.plot(ch + i*5)
        plt.title("Filtered EEG")
        plt.show()




    # ============================================================
    # COMPARE PATIENTS
    # ============================================================
    def show_compare(self):
        self.clear_main()
        tk.Label(self.main, text="Compare Two Patients", font=("Arial", 26)).pack(pady=10)

        container = tk.Frame(self.main, bg="white")
        container.pack(pady=10, fill="x")

        # -------- LEFT COLUMN (A) --------
        left = tk.Frame(container, bg="white")
        left.pack(side="left", padx=60)

        tk.Label(left, text="Paciente A", font=("Arial", 16)).pack()
        self.pA = tk.Entry(left, font=("Arial", 14))
        self.pA.insert(0, "Paciente1")
        self.pA.pack()

        tk.Label(left, text="Group A").pack()
        self.gA = tk.StringVar(value="AD")
        ttk.Combobox(left, textvariable=self.gA, values=["AD", "Healthy"]).pack()

        tk.Label(left, text="State A").pack()
        self.sA = tk.StringVar(value="Eyes_closed")
        ttk.Combobox(left, textvariable=self.sA, values=["Eyes_open", "Eyes_closed"]).pack()

        # -------- RIGHT COLUMN (B) --------
        right = tk.Frame(container, bg="white")
        right.pack(side="right", padx=60)

        tk.Label(right, text="Paciente B", font=("Arial", 16)).pack()
        self.pB = tk.Entry(right, font=("Arial", 14))
        self.pB.insert(0, "Paciente2")
        self.pB.pack()

        tk.Label(right, text="Group B").pack()
        self.gB = tk.StringVar(value="AD")
        ttk.Combobox(right, textvariable=self.gB, values=["AD", "Healthy"]).pack()

        tk.Label(right, text="State B").pack()
        self.sB = tk.StringVar(value="Eyes_open")
        ttk.Combobox(right, textvariable=self.sB, values=["Eyes_open", "Eyes_closed"]).pack()

        # -------- THE COMPARE BUTTON --------
        tk.Button(
            self.main,
            text="COMPARE",
            font=("Arial", 18, "bold"),
            bg="#444",
            fg="white",
            height=2,
            width=15,
            command=self.compare_two
        ).pack(pady=25)




    # ============================================================
    # VIOLIN PLOT WITH CLEAR COLORS
    # ============================================================
    def compare_two(self):
        A = load_subject_raw(self.gA.get(), self.sA.get(), self.pA.get())
        B = load_subject_raw(self.gB.get(), self.sB.get(), self.pB.get())

        if A is None or B is None:
            messagebox.showerror("Error", "One of the patients could not be loaded.")
            return

        # Train temporary classifier
        X, y = build_dataset(self.sA.get())
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y)

        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400)
        )
        clf.fit(X_tr, y_tr)

        fvA = extract_features(preprocess(A)).reshape(1, -1)
        fvB = extract_features(preprocess(B)).reshape(1, -1)

        pA = clf.predict_proba(fvA)[0, 1] * 100
        pB = clf.predict_proba(fvB)[0, 1] * 100

        # Message
        if pA > pB:
            msg = f"{self.pA.get()} has a HIGHER Alzheimer probability by +{pA - pB:.2f}%"
        elif pB > pA:
            msg = f"{self.pB.get()} has a HIGHER Alzheimer probability by +{pB - pA:.2f}%"
        else:
            msg = "Both patients have the same Alzheimer probability."

        messagebox.showinfo(
            "Comparison Result",
            f"{self.pA.get()}: {pA:.2f}%\n"
            f"{self.pB.get()}: {pB:.2f}%\n\n"
            f"{msg}"
        )

        # violin plot
        fvA = fvA.flatten()
        fvB = fvB.flatten()

        df = pd.DataFrame({
            "Feature": np.tile(np.arange(len(fvA)), 2),
            "Value": np.concatenate([fvA, fvB]),
            "Patient": ["A"] * len(fvA) + ["B"] * len(fvB)
        })

        df["Value"] += np.random.normal(0, 0.01, size=len(df))

        plt.figure(figsize=(12, 6))

        palette = {"A": "#FF1493", "B": "#FFD700"}  # pink + yellow

        sns.violinplot(
            x="Feature", y="Value", hue="Patient",
            data=df, split=True, linewidth=4, palette=palette
        )

        sns.stripplot(
            x="Feature", y="Value", hue="Patient",
            data=df, dodge=True, palette=palette,
            size=6, edgecolor="black", linewidth=1
        )

        plt.title("Feature Distribution Comparison")
        plt.tight_layout()
        plt.show()



# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    AlzheimerGUI().mainloop()
