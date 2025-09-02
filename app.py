import streamlit as st
import cv2
import face_recognition
import qrcode
from PIL import Image
import os
import pyzbar.pyzbar as pyzbar
import io
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration - improved accuracy settings ---
PROCESS_EVERY_N_FRAMES = 3  
CAMERA_BUFFER_SIZE = 1 
TOLERANCE = 0.4  # Lower tolerance = more strict matching

# --- Streamlit page config ---
st.set_page_config(page_title="QR + Face Recognition System", layout="wide")

# --- Create folders if they don't exist ---
os.makedirs("hiPic", exist_ok=True)
os.makedirs("qr_codes", exist_ok=True)

# Your exact FaceRecognitionSystem adapted for Streamlit
class StreamlitFaceRecognitionSystem:
    def __init__(self):
        self.reference_encoding = None
        self.reference_image = None
        self.frame_count = 0
        self.last_process_time = time.time()
        self.success_count = 0
        self.last_match_status = None
        self.stable_match_count = 0
        
    def load_reference_image(self, image_path):
        """Load and process reference image - your exact code"""
        if not os.path.exists(image_path):
            return False, f"Reference image not found at: {image_path}"
        
        try:
            self.reference_image = cv2.imread(image_path)
            
            if self.reference_image is None:
                return False, f"Could not load reference image: {image_path}"
            
            # Keep original size, no resize
            reference_image_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
            
            # Get face encoding from reference image
            reference_encodings = face_recognition.face_encodings(reference_image_rgb)
            if len(reference_encodings) == 0:
                return False, "No face found in the reference image!"
            
            self.reference_encoding = reference_encodings[0]
            return True, "Reference encoding loaded successfully"
            
        except Exception as e:
            return False, f"Error loading reference image: {e}"
        
    def initialize_camera(self):
        """Initialize camera with optimal settings - fixed version"""
        video_capture = None
        
        # Try different camera indices if default doesn't work
        for camera_id in [0, 1, 2]:
            try:
                video_capture = cv2.VideoCapture(camera_id)
                if video_capture.isOpened():
                    # Set camera properties
                    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                    video_capture.set(cv2.CAP_PROP_FPS, 30)
                    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Test camera
                    ret, test_frame = video_capture.read()
                    if ret and test_frame is not None:
                        return video_capture, f"Camera {camera_id} initialized successfully"
                    else:
                        video_capture.release()
                        
                if video_capture:
                    video_capture.release()
                    
            except Exception as e:
                if video_capture:
                    video_capture.release()
                continue
        
        return None, "Could not open any camera."
        
    def process_frame(self, frame):
        """Process frame for face recognition with improved accuracy"""
        self.frame_count += 1
        
        # Balanced frame skipping
        current_time = time.time()
        if (self.frame_count % PROCESS_EVERY_N_FRAMES != 0 or 
            current_time - self.last_process_time < 0.15):  # Slightly longer processing time
            # Draw last known results between processing frames
            if hasattr(self, 'last_faces') and self.last_faces:
                for face_info in self.last_faces:
                    top, right, bottom, left, label, color = face_info
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            return frame, False
            
        self.last_process_time = current_time
        
        try:
            # Use larger frame for better accuracy
            medium_frame = cv2.resize(frame, (480, 360))  # Higher resolution
            rgb_medium_frame = cv2.cvtColor(medium_frame, cv2.COLOR_BGR2RGB)
            
            # More thorough face detection
            face_locations = face_recognition.face_locations(
                rgb_medium_frame, 
                model="hog",
                number_of_times_to_upsample=1  # Better face detection
            )
            
            # Scale back up (1.33x scaling for better precision)
            face_locations = [(int(top*4/3), int(right*4/3), int(bottom*4/3), int(left*4/3)) 
                             for (top, right, bottom, left) in face_locations]
            
            # Handle no faces detected
            if not face_locations:
                self.last_faces = []
                self.stable_match_count = max(0, self.stable_match_count - 2)  # Faster decay when no face
                return frame, False
            
            # Get encodings from original frame with higher accuracy
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                face_locations, 
                num_jitters=1,  # Add some jittering for better accuracy
                model="large"   # Use large model for better accuracy
            )
            
            if not face_encodings:
                self.last_faces = []
                self.stable_match_count = max(0, self.stable_match_count - 1)
                return frame, False
            
            match_found = False
            face_info_list = []
            
            # Process detected faces with enhanced matching
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Double validation: tolerance + distance
                matches = face_recognition.compare_faces(
                    [self.reference_encoding], 
                    face_encoding, 
                    tolerance=TOLERANCE
                )
                
                # Calculate face distance for additional validation
                face_distances = face_recognition.face_distance([self.reference_encoding], face_encoding)
                face_distance = face_distances[0]
                
                # Stricter matching criteria
                tolerance_match = matches[0]
                distance_match = face_distance < TOLERANCE
                
                # Both criteria must pass
                match = tolerance_match and distance_match
                match_found = match_found or match
                
                # Enhanced stability logic
                if match:
                    self.success_count += 1
                    self.stable_match_count = min(self.stable_match_count + 2, 15)  # Faster increment
                else:
                    # More gradual decrease to prevent flickering
                    self.stable_match_count = max(0, self.stable_match_count - 1)
                
                # Require more stability for MATCH display (higher threshold)
                display_match = self.stable_match_count >= 5  # Need 5 stable points
                
                # Draw rectangle and label with distance info
                color = (0, 255, 0) if display_match else (0, 0, 255)
                
                if display_match:
                    label = f"MATCH ({face_distance:.3f})"
                else:
                    label = f"NO MATCH ({face_distance:.3f})"
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)  # Thicker rectangle
                
                # Multi-line text for better visibility
                cv2.putText(frame, "MATCH" if display_match else "NO MATCH", 
                           (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Dist: {face_distance:.3f}", 
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Stable: {self.stable_match_count}", 
                           (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Store face info for interpolation frames
                face_info_list.append((top, right, bottom, left, label, color))
            
            # Store results
            self.last_faces = face_info_list
            
            return frame, match_found
            
        except Exception as e:
            # On error, gradually decrease confidence
            self.stable_match_count = max(0, self.stable_match_count - 1)
            self.last_faces = []
            return frame, False

# Initialize the face recognition system in session state
if 'face_system' not in st.session_state:
    st.session_state.face_system = StreamlitFaceRecognitionSystem()

# --- Sidebar Menu ---
st.sidebar.title("📱 QR + Face Recognition System")
menu_option = st.sidebar.selectbox(
    "Choose an option:",
    ["🏠 Home", "📸 Register Student (Upload & Generate QR)", "🔍 Verify Student (Scan QR & Face)"]
)

# --- Home Page ---
if menu_option == "🏠 Home":
    st.title("Welcome to QR + Face Recognition System")
    st.markdown("""
    ## How to use this system:
    
    ### 1. 📸 Register Student
    - Upload a clear photo of the student
    - Enter StudentName_ID (e.g., John_1234)
    - System will generate a QR code for the student
    
    ### 2. 🔍 Verify Student
    - Use camera to scan the student's QR code
    - System will display student information
    - Point face to camera for face verification
    - Get authentication result
    
    **Select an option from the sidebar to get started!**
    """)

# --- Register Student (Upload & Generate QR) ---
elif menu_option == "📸 Register Student (Upload & Generate QR)":
    st.title("📸 Student Registration")
    st.markdown("Upload a student photo and generate their QR code")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Photo")
        uploaded_file = st.file_uploader("Upload student photo", type=["jpg", "jpeg", "png"], key="upload_photo")
        student_input = st.text_input("Enter StudentName_ID (e.g., John_1234)", key="student_input")
        
        if st.button("Generate QR Code", key="generate_qr_btn"):
            if uploaded_file and student_input:
                # Save high-quality photo
                img = Image.open(uploaded_file).convert("RGB")
                img_path = os.path.join("hiPic", f"{student_input}.jpg")
                img.save(img_path, format="JPEG", quality=95)
                st.success(f"✅ High-quality photo saved: {img_path}")

                # Generate QR code (store only name + ID)
                qr_data = student_input
                qr = qrcode.QRCode(version=1, box_size=10, border=4)
                qr.add_data(qr_data)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white")
                qr_path = os.path.join("qr_codes", f"{student_input}.png")
                qr_img.save(qr_path)
                st.success(f"✅ QR code saved: {qr_path}")
            else:
                st.error("Please upload a photo and enter Student ID!")
    
    with col2:
        st.subheader("Preview")
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Photo", width=300)
        
        if student_input and uploaded_file:
            qr_path = os.path.join("qr_codes", f"{student_input}.png")
            if os.path.exists(qr_path):
                st.image(qr_path, caption="Generated QR Code", width=300)

# --- Verify Student (Scan QR & Face) ---
elif menu_option == "🔍 Verify Student (Scan QR & Face)":
    st.title("🔍 Student Verification")
    st.markdown("Scan QR code and verify face for authentication")
    
    # Initialize session state for verification process
    if 'verification_step' not in st.session_state:
        st.session_state.verification_step = 'waiting'
    if 'scanned_student_id' not in st.session_state:
        st.session_state.scanned_student_id = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.verification_step == 'waiting':
            st.info("🎯 Step 1: Click 'Start QR Scanner' and hold QR code in front of camera")
            
        elif st.session_state.verification_step == 'qr_scanned':
            st.success(f"🎯 Step 2: QR Scanned! Student: {st.session_state.scanned_student_id}")
            st.info("📸 Now click 'Start Face Verification' to verify your identity")
            
        elif st.session_state.verification_step == 'verifying':
            st.success(f"🎯 Step 3: Verifying face for student: {st.session_state.scanned_student_id}")
    
    with col2:
        # Show student info if QR is scanned
        if st.session_state.scanned_student_id:
            st.subheader("Student Information")
            st.write(f"**ID:** {st.session_state.scanned_student_id}")
            
            # Show reference photo if available
            img_path = os.path.join("hiPic", f"{st.session_state.scanned_student_id}.jpg")
            if os.path.exists(img_path):
                st.image(img_path, caption="Reference Photo", width=200)
            else:
                st.error("Reference photo not found!")
    
    # Step 1: QR Scanner
    if st.session_state.verification_step == 'waiting':
        if st.button("🎯 Start QR Scanner", key="start_qr_scanner"):
            stframe = st.empty()
            
            # Try multiple camera indices - using your camera logic
            cap = None
            camera_found = False
            
            for camera_index in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            camera_found = True
                            st.info(f"✅ Camera {camera_index} found and working!")
                            break
                        else:
                            cap.release()
                    else:
                        if cap:
                            cap.release()
                except Exception as e:
                    if cap:
                        cap.release()
            
            if not camera_found:
                st.error("❌ Cannot access camera. Please check camera permissions and try again.")
            else:
                scanning_placeholder = st.empty()
                scanning_placeholder.info("📱 Hold QR code steady in front of camera...")
                
                # QR Scanning Loop
                frame_count = 0
                max_frames = 600
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    frame_count += 1
                    
                    if not ret or frame is None:
                        break

                    # Convert to grayscale for better QR detection
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Try decoding on both frames
                    decoded_objects = pyzbar.decode(frame)
                    if not decoded_objects:
                        decoded_objects = pyzbar.decode(gray_frame)
                    
                    # Process detected QR codes
                    for obj in decoded_objects:
                        try:
                            scanned_data = obj.data.decode("utf-8")
                            if scanned_data.strip():
                                # QR code found and scanned
                                st.session_state.scanned_student_id = scanned_data.strip()
                                st.session_state.verification_step = 'qr_scanned'
                                
                                # Load reference image using your system
                                img_path = os.path.join("hiPic", f"{st.session_state.scanned_student_id}.jpg")
                                success, message = st.session_state.face_system.load_reference_image(img_path)
                                
                                if success:
                                    scanning_placeholder.success(f"✅ QR Code Scanned Successfully!")
                                else:
                                    st.error(f"Error: {message}")
                                    st.session_state.verification_step = 'waiting'
                                    st.session_state.scanned_student_id = None
                                
                                cap.release()
                                cv2.destroyAllWindows()
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error decoding QR: {e}")
                    
                    # Add scanning overlay
                    cv2.putText(frame, f"SCANNING QR CODE... {frame_count}/{max_frames}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    stframe.image(frame, channels="BGR")
                
                cap.release()
                cv2.destroyAllWindows()
                
                if st.session_state.verification_step == 'waiting':
                    scanning_placeholder.warning("⏰ QR scanning timeout. Please try again.")
    
    # Step 2: Face Verification
    elif st.session_state.verification_step == 'qr_scanned':
        if st.button("📸 Start Face Verification", key="start_face_verification"):
            st.session_state.verification_step = 'verifying'
            st.rerun()
    
    # Step 3: Face Recognition Process using your exact code
    elif st.session_state.verification_step == 'verifying':
        st.info("Face verification starting...")
        
        if st.button("Reset Verification", key="reset_verification"):
            st.session_state.verification_step = 'waiting'
            st.session_state.scanned_student_id = None
            st.session_state.face_system = StreamlitFaceRecognitionSystem()  # Reset system
            st.rerun()
        
        # Debug: Check if reference encoding is loaded
        if st.session_state.face_system.reference_encoding is None:
            st.error("Reference encoding not loaded. Please scan QR code again.")
        else:
            st.success("Reference encoding loaded successfully")
            
            # Initialize camera using your exact method with debugging
            st.info("Initializing camera...")
            camera, camera_message = st.session_state.face_system.initialize_camera()
            
            if camera is None:
                st.error(f"{camera_message}")
                st.markdown("""
                **Try these solutions:**
                - Close any other apps using the camera (Teams, Zoom, etc.)
                - Refresh the browser page
                - Check browser camera permissions
                - Try a different browser
                """)
            else:
                st.success(f"{camera_message}")
                
                if st.button("Start Face Recognition", key="start_face_rec"):
                    stframe = st.empty()
                    status_placeholder = st.empty()
                    
                    max_frames = 600  # ~20 seconds
                    min_success_matches = 5
                    
                    status_placeholder.info("Point your face towards the camera...")
                    
                    # Reset success count
                    st.session_state.face_system.success_count = 0
                    
                    try:
                        for frame_count in range(max_frames):
                            ret, frame = camera.read()
                            
                            if not ret or frame is None:
                                st.warning(f"Failed to read frame {frame_count}")
                                break
                            
                            # Process frame using your exact system
                            processed_frame, match_found = st.session_state.face_system.process_frame(frame)
                            
                            # Add status overlay
                            cv2.putText(processed_frame, f"SUCCESS COUNT: {st.session_state.face_system.success_count}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"FRAME: {frame_count}/{max_frames}", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Check for authentication success but keep camera running
                            if st.session_state.face_system.success_count >= min_success_matches:
                                status_placeholder.success(f"AUTHENTICATION SUCCESS! Student: {st.session_state.scanned_student_id}")
                                # Don't break here - keep camera running
                            
                            stframe.image(processed_frame, channels="BGR")
                        
                    except Exception as e:
                        st.error(f"Error during face recognition: {e}")
                    finally:
                        # Always cleanup camera
                        if camera:
                            camera.release()
                        cv2.destroyAllWindows()
                    
                    if st.session_state.face_system.success_count < min_success_matches:
                        status_placeholder.warning(f"Verification timeout. Success: {st.session_state.face_system.success_count}/{min_success_matches} needed")
                
                else:
                    # Show a test frame to verify camera is working
                    try:
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None:
                            cv2.putText(test_frame, "CAMERA READY - CLICK START BUTTON", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            st.image(test_frame, channels="BGR", caption="Camera Preview")
                        camera.release()
                    except Exception as e:
                        st.error(f"Camera test failed: {e}")
                        if camera:
                            camera.release()
