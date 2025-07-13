#This script is only for demo purposes. For high performance goals, please make use of C++ inference scripts


import sys
import cv2
import numpy as np
import xir
import vart
import os
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class FrameTimings:
    """Container for frame processing timings"""
    load_time: float = 0.0
    preprocess_time: float = 0.0
    inference_time: float = 0.0
    postprocess_time: float = 0.0
    write_time: float = 0.0
    total_time: float = 0.0

@dataclass
class LetterboxInfo:
    """Container for letterbox transformation parameters"""
    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int

def letterbox_resize(image, target_width=512, target_height=512):
    """
    Resize image using letterbox method to maintain aspect ratio
    Returns resized image and transformation info for reverse operation
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2
    
    # Create padded image
    padded = np.full((target_height, target_width, image.shape[2]), 128, dtype=image.dtype)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # Store transformation info
    letterbox_info = LetterboxInfo(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        original_width=w,
        original_height=h
    )
    
    return padded, letterbox_info

def reverse_letterbox(image, letterbox_info):
    ""
    #Reverse letterbox transformation to restore original dimensions
    
    # Remove padding
    h, w = image.shape[:2]
    new_h = int((h - 2 * letterbox_info.pad_y))
    new_w = int((w - 2 * letterbox_info.pad_x))
    
    # Crop to remove padding
    if letterbox_info.pad_y > 0 or letterbox_info.pad_x > 0:
        cropped = image[letterbox_info.pad_y:letterbox_info.pad_y + new_h,
                      letterbox_info.pad_x:letterbox_info.pad_x + new_w]
    else:
        cropped = image
    
    # Resize back to original dimensions
    restored = cv2.resize(cropped, (letterbox_info.original_width, letterbox_info.original_height))
    
    return restored

def preprocess_image(image_path, fix_scale=128, width=512, height=512):
    """Preprocess image for ZeroDCE++ quantized model on DPU with letterbox"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None, None

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply letterbox resize
    img_resized, letterbox_info = letterbox_resize(img, width, height)

    # Normalize to [0,1] float32
    img_resized = img_resized.astype(np.float32) / 255.0

    # Multiply by fix_scale (usually 128.0 for [-128, 127] range)
    img_resized = img_resized * fix_scale

    # Clip to int8 valid range
    img_resized = np.clip(img_resized, -128, 127).astype(np.int8)

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)  # Shape: (1, H, W, 3)

    return img_resized, letterbox_info

def preprocess_frame(frame, fix_scale=128, width=512, height=512):
    """Preprocess video frame for ZeroDCE++ quantized model on DPU with letterbox"""
    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply letterbox resize
    img_resized, letterbox_info = letterbox_resize(img, width, height)

    # Normalize to [0,1] float32
    img_resized = img_resized.astype(np.float32) / 255.0

    # Multiply by fix_scale (usually 128.0 for [-128, 127] range)
    img_resized = img_resized * fix_scale

    # Clip to int8 valid range
    img_resized = np.clip(img_resized, -128, 127).astype(np.int8)

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)  # Shape: (1, H, W, 3)

    return img_resized, letterbox_info

def frame_loader_thread(cap, frame_queue, max_queue_size=10):
    """Thread to load frames from video file"""
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Block if queue is full
        while frame_queue.qsize() >= max_queue_size:
            time.sleep(0.001)  # Small delay to prevent busy waiting
            
        load_start = time.time()
        frame_queue.put((frame_idx, frame, load_start))
        frame_idx += 1
    
    # Signal end of frames
    frame_queue.put(None)

def processing_thread(thread_id, runner, frame_queue, output_queue, fix_out, 
                     timings_queue, stop_event, original_size):
    """Thread to process frames through the DPU"""
    # Get model parameters
    in_tensor = runner.get_input_tensors()[0]
    out_tensor = runner.get_output_tensors()[0]
    in_shape = tuple(in_tensor.dims)
    out_shape = tuple(out_tensor.dims)
    
    # Create input and output buffers
    input_buffer = np.empty(in_shape, dtype=np.int8)
    output_buffer = np.empty(out_shape, dtype=np.int8)
    
    # Warm-up run
    dummy_frame = np.zeros((*original_size, 3), dtype=np.uint8)
    processed_dummy, _ = preprocess_frame(dummy_frame)
    input_buffer[0] = processed_dummy[0]
    job_id = runner.execute_async([input_buffer], [output_buffer])
    runner.wait(job_id)
    
    while not stop_event.is_set():
        try:
            # Get frame from queue (with timeout)
            frame_data = frame_queue.get(timeout=0.1)
            if frame_data is None:
                break
                
            frame_idx, frame, load_start = frame_data
            timing = FrameTimings()
            
            # Calculate load time
            load_end = time.time()
            timing.load_time = (load_end - load_start) * 1000  # ms
            
            # Preprocess frame
            preprocess_start = time.time()
            processed_frame, letterbox_info = preprocess_frame(frame)
            input_buffer[0] = processed_frame[0]
            preprocess_end = time.time()
            timing.preprocess_time = (preprocess_end - preprocess_start) * 1000  # ms
            
            # Run inference
            inference_start = time.time()
            job_id = runner.execute_async([input_buffer], [output_buffer])
            runner.wait(job_id)
            inference_end = time.time()
            timing.inference_time = (inference_end - inference_start) * 1000  # ms
            
            # Postprocess output
            postprocess_start = time.time()
            output_scale = 1 / (2 ** fix_out)
            output_float = output_buffer.astype(np.float32) * output_scale
            output_img = output_float[0]  # Shape (512, 512, 3)
            output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Reverse letterbox to restore original dimensions
            output_img = reverse_letterbox(output_img, letterbox_info)
            
            postprocess_end = time.time()
            timing.postprocess_time = (postprocess_end - postprocess_start) * 1000  # ms
            
            # Calculate total processing time (excluding write)
            timing.total_time = timing.load_time + timing.preprocess_time + timing.inference_time + timing.postprocess_time
            
            # Send to output queue
            output_queue.put((frame_idx, output_img, timing))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in processing thread {thread_id}: {e}")
            break

def writer_thread(output_queue, out_writer, timings_queue, total_frames):
    """Thread to write processed frames to output video"""
    frames_written = 0
    expected_frame_idx = 0
    frame_buffer = {}  # Buffer to maintain frame order
    
    while frames_written < total_frames:
        try:
            frame_data = output_queue.get(timeout=1.0)
            if frame_data is None:
                break
                
            frame_idx, output_img, timing = frame_data
            
            # Buffer frame if it's not the next expected frame
            if frame_idx != expected_frame_idx:
                frame_buffer[frame_idx] = (output_img, timing)
                continue
            
            # Write current frame
            write_start = time.time()
            out_writer.write(output_img)
            write_end = time.time()
            timing.write_time = (write_end - write_start) * 1000  # ms
            timing.total_time += timing.write_time
            
            # Send timing data
            timings_queue.put(timing)
            
            frames_written += 1
            expected_frame_idx += 1
            
            # Check if buffered frames can be written
            while expected_frame_idx in frame_buffer:
                buffered_img, buffered_timing = frame_buffer.pop(expected_frame_idx)
                
                write_start = time.time()
                out_writer.write(buffered_img)
                write_end = time.time()
                buffered_timing.write_time = (write_end - write_start) * 1000  # ms
                buffered_timing.total_time += buffered_timing.write_time
                
                timings_queue.put(buffered_timing)
                
                frames_written += 1
                expected_frame_idx += 1
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in writer thread: {e}")
            break

def process_video_multithreaded(runners, input_video_path, output_video_path):
    """Process video frame by frame with multiple threads"""
    # Open video capture
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        cap.open(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer (using original resolution)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 30.0, (width, height))
    
    # Get model parameters
    out_tensor = runners[0].get_output_tensors()[0]
    fix_out = out_tensor.get_attr("fix_point")
    
    # Create queues
    frame_queue = queue.Queue()
    output_queue = queue.Queue()
    timings_queue = queue.Queue()
    
    # Create stop event
    stop_event = threading.Event()
    
    print(f"Processing video: {input_video_path}")
    print(f"Total frames: {frame_count if frame_count > 0 else 'unknown'}")
    print(f"Original resolution: {width}x{height}")
    print(f"Original FPS: {fps:.2f}")
    print(f"Using {len(runners)} processing threads")
    print("Starting multi-threaded processing...")
    
    # Start threads
    threads = []
    
    # Frame loader thread
    loader_thread = threading.Thread(
        target=frame_loader_thread,
        args=(cap, frame_queue, 20)  # Max 20 frames in queue
    )
    loader_thread.start()
    threads.append(loader_thread)
    
    # Processing threads
    for i, runner in enumerate(runners):
        proc_thread = threading.Thread(
            target=processing_thread,
            args=(i, runner, frame_queue, output_queue, fix_out, timings_queue, stop_event, (height, width))
        )
        proc_thread.start()
        threads.append(proc_thread)
    
    # Writer thread
    writer_t = threading.Thread(
        target=writer_thread,
        args=(output_queue, out, timings_queue, frame_count if frame_count > 0 else 10000)
    )
    writer_t.start()
    threads.append(writer_t)
    
    # Monitor progress and collect timings
    frames_completed = 0
    total_timings = deque(maxlen=100)  # Keep last 100 frame timings
    last_log_time = time.time()
    start_time = time.time()
    
    while frames_completed < (frame_count if frame_count > 0 else 10000):
        try:
            timing = timings_queue.get(timeout=0.1)
            total_timings.append(timing)
            frames_completed += 1
            
            # Print progress every 2 seconds
            current_time = time.time()
            if current_time - last_log_time > 2.0:
                if total_timings:
                    avg_timing = FrameTimings(
                        load_time=sum(t.load_time for t in total_timings) / len(total_timings),
                        preprocess_time=sum(t.preprocess_time for t in total_timings) / len(total_timings),
                        inference_time=sum(t.inference_time for t in total_timings) / len(total_timings),
                        postprocess_time=sum(t.postprocess_time for t in total_timings) / len(total_timings),
                        write_time=sum(t.write_time for t in total_timings) / len(total_timings),
                        total_time=sum(t.total_time for t in total_timings) / len(total_timings)
                    )
                    
                    # Calculate estimated live FPS for 60 FPS input
                    estimated_fps = 1000 / avg_timing.total_time if avg_timing.total_time > 0 else 0
                    live_fps_ratio = estimated_fps / 60.0 if estimated_fps > 0 else 0
                    
                    print(f"Frames: {frames_completed}/{frame_count if frame_count > 0 else '?'} | "
                          f"Load: {avg_timing.load_time:.1f}ms | "
                          f"Preprocess: {avg_timing.preprocess_time:.1f}ms | "
                          f"Inference: {avg_timing.inference_time:.1f}ms | "
                          f"Postprocess: {avg_timing.postprocess_time:.1f}ms | "
                          f"Write: {avg_timing.write_time:.1f}ms | "
                          f"Total: {avg_timing.total_time:.1f}ms | "
                          f"Est. FPS: {estimated_fps:.1f} | "
                          f"60FPS Live Ratio: {live_fps_ratio:.2f}x")
                    
                last_log_time = current_time
                
        except queue.Empty:
            # Check if all threads are done
            if not any(t.is_alive() for t in threads[:2]):  # loader and processing threads
                break
            continue
        except Exception as e:
            print(f"Error in monitoring: {e}")
            break
    
    # Signal stop and wait for threads to finish
    stop_event.set()
    for t in threads:
        t.join(timeout=5.0)
    
    # Release resources
    cap.release()
    out.release()
    
    # Final summary
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    if total_timings:
        avg_timing = FrameTimings(
            load_time=sum(t.load_time for t in total_timings) / len(total_timings),
            preprocess_time=sum(t.preprocess_time for t in total_timings) / len(total_timings),
            inference_time=sum(t.inference_time for t in total_timings) / len(total_timings),
            postprocess_time=sum(t.postprocess_time for t in total_timings) / len(total_timings),
            write_time=sum(t.write_time for t in total_timings) / len(total_timings),
            total_time=sum(t.total_time for t in total_timings) / len(total_timings)
        )
        
        estimated_fps = 1000 / avg_timing.total_time if avg_timing.total_time > 0 else 0
        live_fps_ratio = estimated_fps / 60.0 if estimated_fps > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"VIDEO PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total frames processed: {frames_completed}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(f"Overall throughput: {frames_completed / total_processing_time:.2f} FPS")
        print(f"Output resolution: {width}x{height} (original)")
        print(f"\nPer-frame timing breakdown (average):")
        print(f"  Load time:        {avg_timing.load_time:.2f} ms")
        print(f"  Preprocess time:  {avg_timing.preprocess_time:.2f} ms")
        print(f"  Inference time:   {avg_timing.inference_time:.2f} ms")
        print(f"  Postprocess time: {avg_timing.postprocess_time:.2f} ms")
        print(f"  Write time:       {avg_timing.write_time:.2f} ms")
        print(f"  Total per frame:  {avg_timing.total_time:.2f} ms")
        print(f"\nEstimated live performance:")
        print(f"  Max sustainable FPS: {estimated_fps:.1f}")
        print(f"  60 FPS live ratio: {live_fps_ratio:.2f}x")
        if live_fps_ratio >= 1.0:
            print(f"  ✓ Can process 60 FPS video in real-time")
        else:
            print(f"  ✗ Cannot process 60 FPS video in real-time")
        print(f"\nOutput video saved to: {output_video_path}")
        print(f"{'='*80}")
    else:
        print("Error: No frames processed")

def worker_thread(runner, input_data, output_data, iterations, thread_id, results):
    """Worker thread function for running inferences"""
    # Warm-up run
    job_id = runner.execute_async([input_data], [output_data])
    runner.wait(job_id)
    
    # Performance measurement
    start_time = time.time()
    
    for i in range(iterations):
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)
    
    end_time = time.time()
    
    # Store results
    results[thread_id] = {
        'total_time': end_time - start_time,
        'latency': (end_time - start_time) / iterations * 1000,  # ms
        'fps': iterations / (end_time - start_time)
    }
    
    # Run one more inference for output (only from first thread)
    if thread_id == 0:
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)
        results['output_data'] = output_data.copy()

def main_image(xmodel_path, input_path, output_path, iterations=100, num_threads=1):
    """Process single image with performance measurements"""
    # Create DPU runners
    graph = xir.Graph.deserialize(xmodel_path)
    root = graph.get_root_subgraph()
    dpus = [c for c in root.toposort_child_subgraph()
            if c.has_attr("device") and c.get_attr("device") == "DPU"]
    
    if not dpus:
        print("No DPU subgraph found!")
        return
    
    runners = []
    for _ in range(num_threads):
        runners.append(vart.Runner.create_runner(dpus[0], "run"))
    
    # Get model parameters
    in_tensor = runners[0].get_input_tensors()[0]
    out_tensor = runners[0].get_output_tensors()[0]
    in_shape = tuple(in_tensor.dims)
    out_shape = tuple(out_tensor.dims)
    fix_in = in_tensor.get_attr("fix_point")
    fix_out = out_tensor.get_attr("fix_point")
    
    print(f"Input shape: {in_shape}, Fix point: {fix_in}")
    print(f"Output shape: {out_shape}, Fix point: {fix_out}")
    print(f"Running with {num_threads} threads, {iterations} iterations per thread")
    
    # Preprocess image
    input_scale = 128
    processed_img, letterbox_info = preprocess_image(input_path, input_scale)
    if processed_img is None:
        return
    
    # Create input/output buffers for each thread
    input_buffers = []
    output_buffers = []
    
    for i in range(num_threads):
        input_data = np.zeros(in_shape, dtype=np.int8)
        output_data = np.zeros(out_shape, dtype=np.int8)
        input_data[0] = processed_img[0]
        input_buffers.append(input_data)
        output_buffers.append(output_data)
    
    # Create worker threads
    threads = []
    results = {}
    
    start_time = time.time()
    
    for i in range(num_threads):
        t = threading.Thread(
            target=worker_thread,
            args=(runners[i], input_buffers[i], output_buffers[i], iterations, i, results)
        )
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    del runners
    
    end_time = time.time()
    
    # Calculate performance metrics
    total_inferences = iterations * num_threads
    total_time = end_time - start_time
    
    # Aggregate thread results
    total_fps = 0
    total_latency = 0
    
    print("\nThread Performance:")
    for i in range(num_threads):
        thread_result = results[i]
        print(f"Thread {i}: {thread_result['fps']:.2f} FPS, "
              f"{thread_result['latency']:.2f} ms")
        total_fps += thread_result['fps']
        total_latency += thread_result['latency']
    
    # Overall metrics
    overall_fps = total_inferences / total_time
    avg_latency = total_latency / num_threads
    
    print("\nOverall Performance:")
    print(f"Total inferences: {total_inferences}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Throughput: {overall_fps:.2f} FPS")
    
    # Save output from first thread
    if 'output_data' in results:
        output_scale = 1 / (2 ** fix_out)
        output_float = results['output_data'].astype(np.float32) * output_scale
        output_img = output_float[0]  # Shape (512, 512, 3)
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Reverse letterbox to restore original dimensions
        output_img = reverse_letterbox(output_img, letterbox_info)
        
        cv2.imwrite(output_path, output_img)
        print(f"\nEnhanced image saved to {output_path}")
        print(f"Output resolution: {letterbox_info.original_width}x{letterbox_info.original_height} (original)")
    else:
        print("Warning: No output image was generated")

def main_video(xmodel_path, input_path, output_path, num_threads=2):
    """Process video file frame by frame with multiple threads"""
    # Create DPU runners
    graph = xir.Graph.deserialize(xmodel_path)
    root = graph.get_root_subgraph()
    dpus = [c for c in root.toposort_child_subgraph()
            if c.has_attr("device") and c.get_attr("device") == "DPU"]
    
    if not dpus:
        print("No DPU subgraph found!")
        return
    
    runners = []
    for _ in range(num_threads):
        runners.append(vart.Runner.create_runner(dpus[0], "run"))
    
    # Process video with multiple threads
    process_video_multithreaded(runners, input_path, output_path)
    
    # Clean up
    for runner in runners:
        del runner

if __name__ == "__main__":
    # Define video extensions
    video_extensions = ['.mp4', '.avi', '.mjpeg','.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    if len(sys.argv) < 4:
        print("Usage: python3 zdce.py <xmodel> <input> <output> [iterations] [threads]")
        print("For images: python3 zdce.py zdce.xmodel input.jpg output.jpg 100 2")
        print("For videos: python3 zdce.py zdce.xmodel input.mp4 output.mp4 [threads]")
        sys.exit(1)
    
    # Extract arguments
    xmodel = sys.argv[1]
    inp = sys.argv[2]
    outp = sys.argv[3]
    
    # Determine if input is video or image
    is_video = any(inp.lower().endswith(ext) for ext in video_extensions)
    
    if is_video:
        # Video processing mode - default to 2 threads
        num_threads = 2
        if len(sys.argv) >= 5:
            try:
                num_threads = int(sys.argv[4])
            except ValueError:
                print("Invalid thread count. Using default 2.")
                
        print(f"Processing video: {inp}")
        print(f"Using {num_threads} thread(s)")
        main_video(xmodel, inp, outp, num_threads)
    else:
        # Image processing mode
        iterations = 100
        num_threads = 1
        
        if len(sys.argv) >= 5:
            try:
                iterations = int(sys.argv[4])
            except ValueError:
                print("Invalid iteration count. Using default 100.")
        
        if len(sys.argv) >= 6:
            try:
                num_threads = int(sys.argv[5])
            except ValueError:
                print("Invalid thread count. Using default 1.")
                
        print(f"Processing image: {inp}")
        main_image(xmodel, inp, outp, iterations, num_threads)
