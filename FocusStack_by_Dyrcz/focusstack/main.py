from pathlib import Path
import argparse
from dataclasses import dataclass
import torch

# local imports
from .io_utils import ensure_dir, list_images, read_images, save_png, save_jpg, save_mask
from .alignment import align_ecc_affine
from .sharpness import argmax_focus_stack, masked_sharpness_score
from .refs import refs_from_base_stack, RefObj
from .assignment import assign_masks_to_refs
from .tracking import tracks_from_assignments, Track
from .corefill import best_frame_for_track, track_mask_core_and_fill
from .union import track_union_mask, best_frame_for_union_mask, clamp_union_to_best
from .composition import hard_paste_then_feather
from .filtering import FilterCfg, filter_masks
from .merging import MergeCfg, merge_similar_masks
from .segmentation import segment_fastsam_single, SegMask

@dataclass
class ObjPack:
    ref_id: int
    track: Track
    best_t: int
    best_s: float
    final_mask: 'np.ndarray'
    area: int
    is_bg: bool

def run(args: argparse.Namespace) -> int:
    import numpy as np
    import cv2 as cv
    try:
        from ultralytics import FastSAM
    except Exception as e:
        raise RuntimeError("Brak ultralytics. Zainstaluj: pip install ultralytics") from e

    in_dir = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    frames_dir = out_dir / "frames"
    base_dir = out_dir / "base"
    objs_dir = out_dir / "objects"

    ensure_dir(frames_dir)
    ensure_dir(base_dir)
    ensure_dir(objs_dir)

    paths = list_images(in_dir, args.pattern)
    imgs = read_images(paths, max_imgs=args.max_imgs)
    h, w = imgs[0].shape[:2]

    if args.align == "ecc":
        imgs_al = align_ecc_affine(imgs, iters=args.ecc_iters, eps=args.ecc_eps, ecc_scale=args.ecc_scale)
    else:
        imgs_al = [im.copy() for im in imgs]

    if args.save_previews:
        ensure_dir(base_dir / "aligned_preview")
        for i, im in enumerate(imgs_al[:min(8, len(imgs_al))]):
            save_jpg(base_dir / "aligned_preview" / f"aligned_{i:03d}.jpg", im, q=90)

    base = argmax_focus_stack(imgs_al, ksize=args.sharp_ksize)
    save_png(out_dir / "base_stack.png", base)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    fp16 = bool(args.fp16) and device == "cuda"

    model = FastSAM(args.fastsam_model)

    fcfg = FilterCfg(
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        nms_iou=args.nms_iou,
        max_masks=args.max_masks,
    )

    mcfg = MergeCfg(
        merge_dist_px=args.merge_dist_px,
        merge_area_ratio_max=args.merge_area_ratio_max,
        merge_aspect_diff=args.merge_aspect_diff,
        merge_bbox_iou_min=args.merge_bbox_iou_min,
        merge_max_iters=args.merge_max_iters,
        dup_iou_thr=args.dup_iou_thr,
        dup_area_sim=args.dup_area_sim,
    )

    refs = refs_from_base_stack(
        base_bgr=base,
        model=model,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        fp16=fp16,
        fcfg=fcfg,
        mcfg=mcfg,
        ref_nms_iou=args.ref_nms_iou,
    )

    if args.save_previews:
        base_masks = [r.mask01 for r in refs]
        base_ids = [r.ref_id for r in refs]
        from .overlay import overlay_masks
        base_overlay = overlay_masks(base, base_masks, ids=base_ids, alpha=0.35)
        save_jpg(base_dir / "seg_base.jpg", base_overlay, q=92)

    for r in refs:
        obj_dir = objs_dir / f"obj_{r.ref_id:03d}"
        ensure_dir(obj_dir)
        ensure_dir(obj_dir / "frame_masks")
        ensure_dir(obj_dir / "previews")
        save_mask(obj_dir / "ref_mask.png", r.mask01)

    assigns_per_frame = []
    grays = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in imgs_al]

    for t, bgr in enumerate(imgs_al):
        frame_dir = frames_dir / f"t{t:03d}"
        ensure_dir(frame_dir)

        seg = segment_fastsam_single(bgr, model, device=device, imgsz=args.imgsz, conf=args.conf, fp16=fp16)
        seg = filter_masks(seg, h, w, fcfg)
        seg = merge_similar_masks(seg, h, w, mcfg)

        amap = assign_masks_to_refs(
            refs=refs,
            frame_masks=seg,
            bbox_gate_iou=args.ref_bbox_gate_iou,
            mask_iou_thr=args.ref_mask_iou_thr,
        )
        assigns_per_frame.append(amap)

        if args.save_previews:
            all_masks = [m.mask01 for m in seg]
            all_ids = list(range(1, len(all_masks) + 1))
            from .overlay import overlay_masks
            prev_all = overlay_masks(bgr, all_masks, ids=all_ids, alpha=0.40)
            save_jpg(frame_dir / "seg_all.jpg", prev_all, q=90)

            ass_masks = [amap[k].mask01 for k in sorted(amap.keys())]
            ass_ids = [k for k in sorted(amap.keys())]
            prev_ass = overlay_masks(bgr, ass_masks, ids=ass_ids, alpha=0.45)
            save_jpg(frame_dir / "seg_assigned.jpg", prev_ass, q=90)

        for ref_id, m in amap.items():
            obj_dir = objs_dir / f"obj_{ref_id:03d}"
            save_mask(obj_dir / "frame_masks" / f"t{t:03d}.png", m.mask01)
            if args.save_previews:
                one = overlay_masks(bgr, [m.mask01], ids=[ref_id], alpha=0.45)
                save_jpg(obj_dir / "previews" / f"t{t:03d}_overlay.jpg", one, q=90)

    tracks = tracks_from_assignments(refs, assigns_per_frame)

    packs = []
    for tr in tracks:
        if not tr.dets:
            continue

        if args.mask_mode == "corefill":
            bt, bs = best_frame_for_track(tr, grays, ksize=args.sharp_ksize, min_area=args.min_area)
            m = track_mask_core_and_fill(
                tr,
                h,
                w,
                best_t=bt,
                core_thr=args.core_thr,
                fill_thr=args.fill_thr,
                max_fill_dist=args.max_fill_dist,
                open_k=args.stab_open_k,
                close_k=args.stab_close_k,
            )
        else:
            m = track_union_mask(
                tr,
                h,
                w,
                open_k=args.stab_open_k,
                close_k=args.stab_close_k,
                skip_area_ratio=args.union_skip_area_ratio,
                do_hole_fill=bool(args.union_hole_fill),
            )
            if int(m.sum()) < args.min_area:
                continue

            bt, bs = best_frame_for_union_mask(
                tr,
                grays,
                union_mask01=m,
                ksize=args.sharp_ksize,
                min_area=args.min_area,
            )

            if bool(args.union_clamp):
                best_mask = _det_mask_at_t(tr, bt)
                if best_mask is not None and int(best_mask.sum()) > 0:
                    m = clamp_union_to_best(
                        union_mask01=m,
                        best_mask01=best_mask,
                        kernel=args.clamp_kernel,
                        iters=args.clamp_iters,
                    )
                    if int(m.sum()) < args.min_area:
                        continue

        area = int(m.sum())
        if area < args.min_area:
            continue

        is_bg = (area / float(h * w)) >= args.bg_area_ratio
        packs.append(ObjPack(ref_id=tr.track_id, track=tr, best_t=bt, best_s=bs, final_mask=m, area=area, is_bg=is_bg))

    for p in packs:
        obj_dir = objs_dir / f"obj_{p.ref_id:03d}"
        save_mask(obj_dir / "final_mask.png", p.final_mask)
        if args.save_previews:
            src = imgs_al[p.best_t]
            from .overlay import overlay_masks
            prev = overlay_masks(src, [p.final_mask], ids=[p.ref_id], alpha=0.45)
            save_jpg(obj_dir / "previews" / f"best_t{p.best_t:03d}_finalmask.jpg", prev, q=92)

    bg = sorted([p for p in packs if p.is_bg], key=lambda x: (x.best_s, x.area), reverse=True)
    fg = sorted([p for p in packs if not p.is_bg], key=lambda x: (x.best_s, x.area), reverse=True)

    lines = []
    for p in bg:
        lines.append(f"BG id={p.ref_id} len={len(p.track.dets)} best_t={p.best_t} best_score={p.best_s:.4f} area={p.area}")
    for p in fg:
        lines.append(f"FG id={p.ref_id} len={len(p.track.dets)} best_t={p.best_t} best_score={p.best_s:.4f} area={p.area}")
    (out_dir / "tracks_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    final = base.copy()

    for p in bg:
        src = imgs_al[p.best_t]
        final = hard_paste_then_feather(final, src, p.final_mask, feather=args.feather_bg)

    for p in fg:
        src = imgs_al[p.best_t]
        final = hard_paste_then_feather(final, src, p.final_mask, feather=args.feather_fg)

    save_png(out_dir / "final.png", final)
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("focusstack_object_pipeline_fastsam_gpu")

    p.add_argument("--input", required=True)
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_imgs", type=int, default=0)

    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--fp16", type=int, default=1)

    p.add_argument("--align", choices=["none", "ecc"], default="ecc")
    p.add_argument("--ecc_scale", type=float, default=0.5)
    p.add_argument("--ecc_iters", type=int, default=80)
    p.add_argument("--ecc_eps", type=float, default=1e-6)

    p.add_argument("--save_previews", type=int, default=1)

    p.add_argument("--fastsam_model", default="FastSAM-s.pt")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--conf", type=float, default=0.20)

    p.add_argument("--min_area", type=int, default=200)
    p.add_argument("--max_area_ratio", type=float, default=0.98)
    p.add_argument("--nms_iou", type=float, default=0.60)
    p.add_argument("--max_masks", type=int, default=120)

    p.add_argument("--merge_dist_px", type=float, default=45.0)
    p.add_argument("--merge_area_ratio_max", type=float, default=3.0)
    p.add_argument("--merge_aspect_diff", type=float, default=0.9)
    p.add_argument("--merge_bbox_iou_min", type=float, default=0.02)
    p.add_argument("--merge_max_iters", type=int, default=3)
    p.add_argument("--dup_iou_thr", type=float, default=0.80)
    p.add_argument("--dup_area_sim", type=float, default=0.80)

    p.add_argument("--ref_nms_iou", type=float, default=0.85)
    p.add_argument("--ref_bbox_gate_iou", type=float, default=0.03)
    p.add_argument("--ref_mask_iou_thr", type=float, default=0.20)

    p.add_argument("--sharp_ksize", type=int, default=3)

    # stabilizacja / corefill (zostawione)
    p.add_argument("--core_thr", type=float, default=0.82)
    p.add_argument("--fill_thr", type=float, default=0.62)
    p.add_argument("--max_fill_dist", type=int, default=18)
    p.add_argument("--stab_open_k", type=int, default=3)
    p.add_argument("--stab_close_k", type=int, default=5)

    # NOWE: tryb maski
    p.add_argument("--mask_mode", choices=["union", "corefill"], default="union")

    # union settings
    p.add_argument("--union_skip_area_ratio", type=float, default=0.90)
    p.add_argument("--union_hole_fill", type=int, default=0)

    # clamp union do okolicy best klatki
    p.add_argument("--union_clamp", type=int, default=1)
    p.add_argument("--clamp_kernel", type=int, default=11)
    p.add_argument("--clamp_iters", type=int, default=2)

    # kompozycja
    p.add_argument("--bg_area_ratio", type=float, default=0.35)
    p.add_argument("--feather_bg", type=int, default=13)
    p.add_argument("--feather_fg", type=int, default=21)

    return p

if __name__ == "__main__":
    import sys
    sys.exit(run(build_parser().parse_args()))
