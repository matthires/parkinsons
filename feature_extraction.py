import os
import numpy as np
import pandas as pd
import librosa
from collections import defaultdict
import surfboard.feature_extraction as fe
from surfboard.sound import Waveform

DATA_SAVE_DIR = "data"
DEFAULT_SR = 16000

# Use this many milliseconds from start and end of each audio sample.
# Set to None to disable onset+offset cropping.
ONSET_OFFSET_MS = 50


class FeatureExtractor:
    def __init__(self, sample_rate: int = DEFAULT_SR, onset_offset_ms: float = ONSET_OFFSET_MS):
        """
        Parameters
        ----------
        sample_rate : int
            Target sampling rate for loading audio.
        onset_offset_ms : float or None
            If not None, each sample will be reduced to a concatenation of:
            - onset segment of length onset_offset_ms (from the start)
            - offset segment of length onset_offset_ms (from the end)
            in milliseconds, after voiced-trimming + pre-emphasis.
        """
        self.sample_rate = sample_rate
        self.onset_offset_ms = onset_offset_ms

    # ---------- utilities ----------
    def _safe_preemphasis(self, y):
        if y is None or len(y) == 0:
            return y
        return librosa.effects.preemphasis(y)

    def get_voiced_samples(self, sample, top_db: int = 15):
        """
        Extract voiced (non-silent) segments and concatenate them.
        Falls back to the original sample if nothing is detected.
        """
        segments = librosa.effects.split(sample, top_db=top_db)
        if len(segments) == 0:
            return sample
        return np.concatenate([sample[s:e] for (s, e) in segments])

    def aggregate_extracted_features(self, feature_set: pd.DataFrame, statistics):
        """
        Aggregate time-dependent features (with stat suffixes) to one value per row.
        """
        out_rows = []
        for _, row in feature_set.iterrows():
            agg = {}
            groups = defaultdict(lambda: defaultdict(list))
            for key, val in row.items():
                matched = False
                for stat in statistics:
                    if f"_{stat}_" in key or key.endswith(f"_{stat}"):
                        base = key.split(f"_{stat}")[0]
                        groups[base][stat].append(val)
                        matched = True
                        break
                if not matched:
                    agg[key] = val
            # aggregate segment based features
            for base, stats_dict in groups.items():
                for stat, values in stats_dict.items():
                    vals = np.asarray(values, dtype=float)
                    if stat == "mean":
                        agg[f"{base}_{stat}"] = np.mean(vals)
                    elif stat == "std":
                        agg[f"{base}_{stat}"] = np.std(vals)
                    elif stat == "min":
                        agg[f"{base}_{stat}"] = np.min(vals)
                    elif stat == "max":
                        agg[f"{base}_{stat}"] = np.max(vals)
                    elif stat == "first_quartile":
                        agg[f"{base}_{stat}"] = np.percentile(vals, 25)
                    elif stat == "second_quartile":
                        agg[f"{base}_{stat}"] = np.percentile(vals, 50)
                    elif stat == "third_quartile":
                        agg[f"{base}_{stat}"] = np.percentile(vals, 75)
                    elif stat == "percentile_1":
                        agg[f"{base}_{stat}"] = np.percentile(vals, 1)
                    elif stat == "percentile_99":
                        agg[f"{base}_{stat}"] = np.percentile(vals, 99)
                    elif stat == "q2_q1_range":
                        q1, q2 = np.percentile(vals, 25), np.percentile(vals, 50)
                        agg[f"{base}_{stat}"] = float(q2 - q1)
                    elif stat == "q3_q2_range":
                        q2, q3 = np.percentile(vals, 50), np.percentile(vals, 75)
                        agg[f"{base}_{stat}"] = float(q3 - q2)
                    elif stat == "q3_q1_range":
                        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
                        agg[f"{base}_{stat}"] = float(q3 - q1)
                    else:
                        raise ValueError(f"Unknown statistic: {stat}")
            out_rows.append(agg)
        return pd.DataFrame(out_rows)

    def _apply_onset_offset(self, y: np.ndarray) -> np.ndarray:
        """
        Optionally replace the signal with onset+offset segments of fixed duration.

        - If self.onset_offset_ms is None -> return y unchanged.
        - If the sample is shorter than 2 * window -> fall back to original y.

        Returns
        -------
        np.ndarray
            Either the original y, or concatenated onset+offset.
        """
        if self.onset_offset_ms is None:
            return y

        win_samples = int(self.sample_rate * self.onset_offset_ms / 1000.0)
        if win_samples <= 0:
            return y

        n = len(y)
        if n < 2 * win_samples:
            # Too short: keep the whole voiced sample
            return y

        onset = y[:win_samples]
        offset = y[-win_samples:]
        return np.concatenate([onset, offset])

    def _iter_audio_files(self, root_path: str, tasks: list, exts=(".wav",)):
        """
        Yield (file_path, subject, task) for files under:
            root/subject/task/*.wav
        If `tasks` is provided, only include those task folder names.
        """
        if not os.path.isdir(root_path):
            return
        for subject in sorted(os.listdir(root_path)):
            subj_dir = os.path.join(root_path, subject)
            if not os.path.isdir(subj_dir):
                continue
            for task in sorted(os.listdir(subj_dir)):
                if tasks and task not in tasks:
                    continue
                task_dir = os.path.join(subj_dir, task)
                if not os.path.isdir(task_dir):
                    continue
                for fname in sorted(os.listdir(task_dir)):
                    if fname.lower().endswith(exts):
                        yield os.path.join(task_dir, fname), subject, task

    def extract_features(
        self,
        root_paths,
        tasks: list = None,
        save: bool = True,
        save_name: str = "features.csv",
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """
        Extract features from 1 or 2 dataset roots in a memory-safe, chunked way.

        - If ONE path is provided: unlabeled (no 'label' column).
        - If TWO paths are provided: first = class 0, second = class 1 (labels added).

        Parameters
        ----------
        root_paths : str or list[str]
            One or two root directories with layout root/subject/task/*.wav.
        tasks : list[str] or None
            Optional subset of task folder names to include.
        save : bool
            If True, append chunks to csv/<save_name>.
        save_name : str
            CSV filename (inside DATA_SAVE_DIR).
        chunk_size : int
            Number of audio files to process in one surfboard.extract_features() call.

        Returns
        -------
        pd.DataFrame
            All extracted features concatenated.
        """
        if isinstance(root_paths, str):
            roots = [root_paths]
        else:
            roots = list(root_paths)

        if len(roots) not in (1, 2):
            raise ValueError("Provide 1 (unlabeled) or 2 (class0, class1) root paths.")

        labeled = len(roots) == 2

        components = [
            "log_melspec",
            "morlet_cwt",
            "f0_statistics",
            "loudness",
            "rms",
            "f0_contour",
            "formants",
            "lpc",
            "dfa",
            "hnr",
            "shimmers",
            "jitters",
            "ppe",
            "log_energy",
            "intensity",
            "zerocrossing",
            "mfcc",
            "chroma_stft",
            "chroma_cqt",
            "chroma_cens",
            "formants_slidingwindow",
            "magnitude_spectrum",
            "bark_spectrogram",
            "spectral_slope",
            "spectral_flux",
            "spectral_entropy",
            "spectral_centroid",
            "spectral_spread",
            "lsf",
        ]
        statistics = [
            "mean",
            "min",
            "max",
            "std",
            "first_quartile",
            "second_quartile",
            "third_quartile",
            "percentile_1",
            "percentile_99",
            "q2_q1_range",
            "q3_q2_range",
            "q3_q1_range",
        ]

        if save:
            os.makedirs(DATA_SAVE_DIR, exist_ok=True)
            out_path = os.path.join(DATA_SAVE_DIR, save_name)
            # if file exists, overwrite (first remove)
            if os.path.exists(out_path):
                os.remove(out_path)
        else:
            out_path = None

        all_chunks = []  # only used if save=False

        header_written = False

        for root_idx, root in enumerate(roots):
            curr_label = root_idx if labeled else None  # 0 for first, 1 for second

            # chunk buffers
            waveforms = []
            meta_subjects = []
            meta_tasks = []
            meta_labels = []

            for fpath, subject, task in self._iter_audio_files(root, tasks):
                # load & preprocess audio
                y, _ = librosa.load(fpath, sr=self.sample_rate, mono=True)
                y = self.get_voiced_samples(y, top_db=15)
                y = librosa.effects.preemphasis(y)

                # optionally keep only onset+offset segments
                y = self._apply_onset_offset(y)

                waveforms.append(Waveform(signal=y, sample_rate=self.sample_rate))
                meta_subjects.append(subject)
                meta_tasks.append(task)
                if labeled:
                    meta_labels.append(curr_label)

                # process chunk
                if len(waveforms) >= chunk_size:
                    feat_chunk = fe.extract_features(waveforms, components, statistics)
                    feat_chunk = self.aggregate_extracted_features(feat_chunk, statistics)

                    feat_chunk = feat_chunk.assign(
                        subject=meta_subjects,
                        task=meta_tasks,
                    )
                    if labeled:
                        feat_chunk = feat_chunk.assign(label=meta_labels)

                    if save:
                        mode = "a" if header_written else "w"
                        feat_chunk.to_csv(out_path, mode=mode, header=not header_written, index=False)
                        header_written = True
                    else:
                        all_chunks.append(feat_chunk)

                    # reset buffers
                    waveforms = []
                    meta_subjects = []
                    meta_tasks = []
                    meta_labels = []

            # flush last partial chunk for this root
            if waveforms:
                feat_chunk = fe.extract_features(waveforms, components, statistics)
                feat_chunk = self.aggregate_extracted_features(feat_chunk, statistics)

                feat_chunk = feat_chunk.assign(
                    subject=meta_subjects,
                    task=meta_tasks,
                )
                if labeled:
                    feat_chunk = feat_chunk.assign(label=meta_labels)

                if save:
                    mode = "a" if header_written else "w"
                    feat_chunk.to_csv(out_path, mode=mode, header=not header_written, index=False)
                    header_written = True
                else:
                    all_chunks.append(feat_chunk)

        if not header_written and save:
            # no data at all
            raise RuntimeError(f"No .wav files found under: {roots}")

        if save:
            # read back concatenated CSV (for API compatibility)
            return pd.read_csv(out_path)
        else:
            if not all_chunks:
                raise RuntimeError(f"No .wav files found under: {roots}")
            return pd.concat(all_chunks, axis=0, ignore_index=True)


# -------------------------
# Example usage
# -------------------------
# fx = FeatureExtractor(sample_rate=16000)

# 1) Unlabeled evaluation (ONE path):
# eval_root = "/data/eval_root"  # <root>/<subject>/<task>/*.wav
# df_eval = fx.extract_features(eval_root, tasks=None, save=True, save_name="eval_features.csv")
# assert "label" not in df_eval.columns

# 2) Labeled training (TWO paths):
# class0_root = "/data/train/class0_root"  # label 0
# class1_root = "/data/train/class1_root"  # label 1
# df_train = fx.extract_features([class0_root, class1_root], tasks=None, save=True, save_name="train_features.csv")
# assert "label" in df_train.columns
