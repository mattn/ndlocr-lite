from PIL import Image
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List


def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class PARSEQ:
    def __init__(self,
                 model_path: str,
                 charlist: [str],
                 original_size: Tuple[int, int] = (384, 32),
                 device: str = "CPU") -> None:
        self.model_path = model_path
        self.charlist = charlist

        self.device = device
        self.image_width, self.image_height = original_size
        self.create_session()

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        #opt_session.enable_cpu_mem_arena = False
        #opt_session.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        #opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() == "cpu":
            opt_session.intra_op_num_threads = 1
            opt_session.inter_op_num_threads = 1
        elif self.device.casefold() == "cuda":
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        session = onnxruntime.InferenceSession(self.model_path,opt_session, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)
        if pil_image.height>pil_image.width:
            pil_image =pil_image.transpose(Image.ROTATE_90)
        pil_resized = pil_image.resize((self.input_width, self.input_height))

        resized = np.array(pil_resized, dtype=np.float32)
        resized = resized[:,:,::-1]
        input_image = resized / 255.0
        input_image = 2.0*(input_image-0.5)
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _preprocess_no_rotation(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)
        pil_resized = pil_image.resize((self.input_width, self.input_height))
        resized = np.array(pil_resized, dtype=np.float32)
        resized = resized[:,:,::-1]
        input_image = resized / 255.0
        input_image = 2.0*(input_image-0.5)
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _read_with_confidence(self, img: np.ndarray, rotate: bool = True) -> Tuple[str, List[float]]:
        if img is None:
            return "", []
        if rotate:
            input_tensor = self.preprocess(img)
        else:
            input_tensor = self._preprocess_no_rotation(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        probs = _softmax(outputs, axis=2)
        indices = np.argmax(probs, axis=2)[0]
        max_probs = np.max(probs, axis=2)[0]
        stop_idx = np.where(indices == 0)[0]
        end_pos = stop_idx[0] if stop_idx.size > 0 else len(indices)
        char_indices = indices[:end_pos].tolist()
        confidences = max_probs[:end_pos].tolist()
        text = "".join([self.charlist[i - 1] for i in char_indices])
        return text, confidences

    @staticmethod
    def _segment_blocks(img: np.ndarray, min_gap: int = 5) -> List[Tuple[int, int]]:
        if img.ndim == 3:
            gray = np.mean(img, axis=2).astype(np.uint8)
        else:
            gray = img
        threshold = int(np.mean(gray))
        binary = (gray < threshold).astype(np.int32)
        proj = np.sum(binary, axis=1)
        is_ink = proj > 0
        blocks: List[Tuple[int, int]] = []
        in_block = False
        start = 0
        for y in range(len(is_ink)):
            if is_ink[y] and not in_block:
                start = y
                in_block = True
            elif not is_ink[y] and in_block:
                blocks.append((start, y))
                in_block = False
        if in_block:
            blocks.append((start, len(is_ink)))
        merged: List[Tuple[int, int]] = []
        for b in blocks:
            if merged and b[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], b[1])
            else:
                merged.append(b)
        return merged

    @staticmethod
    def _count_horizontal_components(segment: np.ndarray) -> int:
        if segment.ndim == 3:
            gray = np.mean(segment, axis=2).astype(np.uint8)
        else:
            gray = segment
        threshold = int(np.mean(gray))
        binary = (gray < threshold).astype(np.int32)
        col_sum = np.sum(binary, axis=0)
        if col_sum.max() == 0:
            return 0
        ink_threshold = col_sum.max() * 0.10
        is_ink = col_sum > ink_threshold
        components = 0
        in_component = False
        for v in is_ink:
            if v and not in_component:
                components += 1
                in_component = True
            elif not v:
                in_component = False
        return components

    def _detect_and_fix_tatechuyoko(self, img: np.ndarray) -> str:
        h, w = img.shape[:2]
        full_text, full_conf = self._read_with_confidence(img, rotate=True)
        if not full_text:
            return full_text
        blocks = self._segment_blocks(img)
        if not blocks:
            return full_text
        patches: List[Tuple[int, int, str, List[float]]] = []
        for y_start, y_end in blocks:
            block_height = y_end - y_start
            margin = max(2, int(block_height * 0.1))
            y0 = max(0, y_start - margin)
            y1 = min(h, y_end + margin)
            block_img = img[y0:y1, :, :] if img.ndim == 3 else img[y0:y1, :]
            if self._count_horizontal_components(block_img) < 2:
                continue
            if block_img.ndim == 2:
                block_img = np.stack([block_img] * 3, axis=-1)
            seg_text, seg_conf = self._read_with_confidence(block_img, rotate=False)
            if not seg_text:
                continue
            ratio_start = y_start / h
            ratio_end = y_end / h
            n_chars = len(full_text)
            char_start = max(0, int(round(ratio_start * n_chars)))
            char_end = min(n_chars, int(round(ratio_end * n_chars)))
            if char_end <= char_start:
                char_end = char_start + 1
            if char_start < len(full_conf):
                region_conf = full_conf[char_start:char_end]
                avg_full_conf = np.mean(region_conf) if len(region_conf) > 0 else 0.0
            else:
                avg_full_conf = 0.0
            avg_seg_conf = np.mean(seg_conf) if seg_conf else 0.0
            if avg_seg_conf > avg_full_conf:
                patches.append((char_start, char_end, seg_text, seg_conf))
        if not patches:
            return full_text
        patches.sort(key=lambda p: p[0])
        resolved: List[Tuple[int, int, str, List[float]]] = []
        for patch in patches:
            if resolved and patch[0] < resolved[-1][1]:
                prev = resolved[-1]
                if np.mean(patch[3]) > np.mean(prev[3]):
                    resolved[-1] = patch
            else:
                resolved.append(patch)
        result_parts: List[str] = []
        pos = 0
        for char_start, char_end, seg_text, _ in resolved:
            if pos < char_start:
                result_parts.append(full_text[pos:char_start])
            result_parts.append(seg_text)
            pos = char_end
        if pos < len(full_text):
            result_parts.append(full_text[pos:])
        return "".join(result_parts)

    def read(self, img: np.ndarray) -> List:
        if img is None:
            return None
        h, w = img.shape[:2]
        if h > w:
            return self._detect_and_fix_tatechuyoko(img)
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        indices = np.argmax(outputs, axis=2)[0]
        stop_idx = np.where(indices == 0)[0]
        end_pos = stop_idx[0] if stop_idx.size > 0 else len(indices)
        resval = indices[:end_pos].tolist()
        resstr = "".join([self.charlist[i - 1] for i in resval])
        return resstr