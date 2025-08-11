import speech_recognition as sr
import pyttsx3
import webbrowser
import datetime
import os
import subprocess
import psutil
import pywhatkit
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import winreg
import glob
import time
import random
from google import genai
import os

class ALPHAGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ALPHA - AI Voice Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg='#0a0a0a')
        
        # Initialize speech components
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.is_running = False
        self.wake_word_mode = False
        self.wake_words = ["alpha", "hey alpha", "ok alpha"]
        
        # Message composition state
        self.waiting_for_message = False
        self.current_contact = ""
        self.message_type = ""
        
        # User preferences
        self.user_name = ""
        self.first_run = True
        
        # Gemini API Configuration
        try:
            api_key = os.environ.get("AIzaSyCQSmV6lufRGU3ISXoKb7xn0o5xJG7gUiI")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            self.ai_enabled = True
        except Exception as e:
            self.ai_enabled = False
            print(f"Gemini initialization error: {str(e)}. AI mode disabled.")
        
        self.conversation_history = []
        
        # Software paths
        self.software_paths = {
            "spotify": [r"C:\Users\{}\AppData\Roaming\Spotify\Spotify.exe".format(os.getlogin()),
                       r"C:\Program Files\Spotify\Spotify.exe"],
            "obs studio": [r"C:\Program Files\obs-studio\bin\64bit\obs64.exe"],
            "bluestacks": [r"C:\Program Files\BlueStacks\Bluestacks.exe",
                          r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"],
            "arduino": [r"C:\Program Files (x86)\Arduino\arduino.exe",
                       r"C:\Program Files\Arduino\arduino.exe"],
            "notepad": [r"C:\Windows\System32\notepad.exe"],
            "calculator": [r"C:\Windows\System32\calc.exe"],
            "chrome": [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                      r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
            "firefox": [r"C:\Program Files\Mozilla Firefox\firefox.exe",
                       r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"],
            "edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
            "vlc": [r"C:\Program Files\VideoLAN\VLC\vlc.exe",
                   r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"],
            "steam": [r"C:\Program Files (x86)\Steam\steam.exe",
                     r"C:\Program Files\Steam\steam.exe"],
            "discord": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge_proxy.exe".format(os.getlogin())],
            "telegram": [r"C:\Users\{}\AppData\Roaming\Telegram Desktop\Telegram.exe".format(os.getlogin())],
            "whatsapp": [
                os.path.expanduser(r"~\AppData\Local\WhatsApp\WhatsApp.exe"),
                r"C:\Users\{}\AppData\Local\WhatsApp\WhatsApp.exe".format(os.getlogin()),
                r"C:\Program Files\WhatsApp\WhatsApp.exe",
                r"C:\Program Files (x86)\WhatsApp\WhatsApp.exe",
                os.path.expanduser(r"~\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\WhatsApp.exe")
            ],
            "photoshop": [r"C:\Program Files\Adobe\Adobe Photoshop 2023\Photoshop.exe",
                         r"C:\Program Files\Adobe\Adobe Photoshop 2022\Photoshop.exe"],
            "word": [r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"],
            "excel": [r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"],
            "powerpoint": [r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"],
            "vscode": [r"C:\Users\{}\AppData\Local\Programs\Microsoft VS Code\Code.exe".format(os.getlogin()),
                      r"C:\Program Files\Microsoft VS Code\Code.exe"]
        }
        
        # WhatsApp contacts
        self.whatsapp_contacts = {
            "mom": "+1234567890",     # Replace with real phone numbers
            "dad": "++1234567890",
            "rushi": "++1234567890",
            "sarah": "+1234567890",
            "brother": "+1234567890",
            "sister": "+1234567890",
            "friend": "+1234567890",
        }
        
        # File search directories
        self.search_directories = [
            os.path.expanduser("~"),
            r"C:\Users\{}\Desktop".format(os.getlogin()),
            r"C:\Users\{}\Documents".format(os.getlogin()),
            r"C:\Users\{}\Downloads".format(os.getlogin()),
            r"C:\Users\{}\Pictures".format(os.getlogin()),
            r"C:\Users\{}\Videos".format(os.getlogin()),
            r"C:\Users\{}\Music".format(os.getlogin())
        ]
        
        # Basic command keywords
        self.basic_commands = [
            "time", "date", "open", "close", "stop", "search", "play", "whatsapp", 
            "send message", "message", "file", "sleep", "wake", 
            "disable wake", "always listen", "list contacts", "show contacts",
            "exit", "bye", "shutdown", "okay", "no", "mind", "cancel", "my name is",
            "what is my name", "who am i", "how are you", "thank you", "thanks",
            "good morning", "good afternoon", "good evening", "good night",
            "i love you", "love you", "tell me a joke", "joke"
        ]
        
        # Greeting variations
        self.greetings = [
            "Hello! How are you doing today?",
            "Hi there! What can I help you with?",
            "Good to see you! How has your day been?",
            "Hello! I'm here and ready to assist you!",
            "Hi! Hope you're having a great day!",
            "Good day! What would you like me to help you with?",
            "Hello! I'm excited to help you today!",
            "Hi there! Ready to make your day easier?"
        ]
        
        self.setup_gui()
        
        # No additional visual systems in original build
        
    def setup_gui(self):
        """Setup the GUI"""
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title_label = tk.Label(
            main_frame, 
            text="A.L.P.H.A", 
            font=("Arial", 28, "bold"), 
            fg='#00ffff', 
            bg='#0a0a0a'
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(
            main_frame, 
            text="Advanced Learning & Processing Hub Assistant with AI", 
            font=("Arial", 12), 
            fg='#ffffff', 
            bg='#0a0a0a'
        )
        subtitle_label.pack(pady=(0, 20))
        
        status_frame = tk.Frame(main_frame, bg='#0a0a0a')
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.status_label = tk.Label(
            status_frame, 
            text="● OFFLINE ●", 
            font=("Arial", 14, "bold"), 
            fg='#ff4444', 
            bg='#0a0a0a'
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.ai_status_label = tk.Label(
            status_frame, 
            text="🤖 AI: READY" if self.ai_enabled else "🤖 AI: DISABLED",
            font=("Arial", 12), 
            fg='#00ffaa' if self.ai_enabled else '#888888',
            bg='#0a0a0a'
        )
        self.ai_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.wake_mode_label = tk.Label(
            status_frame, 
            text="WAKE MODE: OFF", 
            font=("Arial", 12), 
            fg='#00ff00', 
            bg='#0a0a0a'
        )
        self.wake_mode_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.message_status_label = tk.Label(
            status_frame, 
            text="", 
            font=("Arial", 12), 
            fg='#ff00ff', 
            bg='#0a0a0a'
        )
        self.message_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.listening_label = tk.Label(
            status_frame, 
            text="", 
            font=("Arial", 12), 
            fg='#ffff00', 
            bg='#0a0a0a'
        )
        self.listening_label.pack(side=tk.RIGHT)

        # Original UI had no extra visual canvases
        
        control_frame = tk.Frame(main_frame, bg='#0a0a0a')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_button = tk.Button(
            control_frame,
            text="🔴 START ALPHA",
            font=("Arial", 12, "bold"),
            bg='#004400',
            fg='#00ff00',
            activebackground='#006600',
            activeforeground='#ffffff',
            border=0,
            padx=20,
            pady=10,
            command=self.toggle_ALPHA
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.wake_toggle_button = tk.Button(
            control_frame,
            text="🌙 TOGGLE WAKE MODE",
            font=("Arial", 12, "bold"),
            bg='#442200',
            fg='#ffaa00',
            activebackground='#664400',
            activeforeground='#ffffff',
            border=0,
            padx=20,
            pady=10,
            command=self.toggle_wake_mode
        )
        self.wake_toggle_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.manual_listen_button = tk.Button(
            control_frame,
            text="🎤 MANUAL LISTEN",
            font=("Arial", 12, "bold"),
            bg='#444400',
            fg='#ffff00',
            activebackground='#666600',
            activeforeground='#ffffff',
            border=0,
            padx=20,
            pady=10,
            command=self.manual_listen,
            state=tk.DISABLED
        )
        self.manual_listen_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.ai_toggle_button = tk.Button(
            control_frame,
            text="🤖 TOGGLE AI",
            font=("Arial", 12, "bold"),
            bg='#004444',
            fg='#00ffff',
            activebackground='#006666',
            activeforeground='#ffffff',
            border=0,
            padx=20,
            pady=10,
            command=self.toggle_ai
        )
        self.ai_toggle_button.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_button = tk.Button(
            control_frame,
            text="🗑️ CLEAR LOG",
            font=("Arial", 12, "bold"),
            bg='#440044',
            fg='#ff00ff',
            activebackground='#660066',
            activeforeground='#ffffff',
            border=0,
            padx=20,
            pady=10,
            command=self.clear_log
        )
        clear_button.pack(side=tk.RIGHT)
        
        console_label = tk.Label(
            main_frame, 
            text="SYSTEM LOG", 
            font=("Arial", 12, "bold"), 
            fg='#00ffff', 
            bg='#0a0a0a'
        )
        console_label.pack(anchor=tk.W)
        
        self.console = scrolledtext.ScrolledText(
            main_frame,
            height=15,
            font=("Consolas", 10),
            bg='#1a1a1a',
            fg='#00ff00',
            insertbackground='#00ff00',
            selectbackground='#004400',
            wrap=tk.WORD
        )
        self.console.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        input_frame = tk.Frame(main_frame, bg='#0a0a0a')
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        input_label = tk.Label(
            input_frame, 
            text="MANUAL COMMAND:", 
            font=("Arial", 10, "bold"), 
            fg='#ffffff', 
            bg='#0a0a0a'
        )
        input_label.pack(anchor=tk.W)
        
        self.command_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            bg='#2a2a2a',
            fg='#ffffff',
            insertbackground='#ffffff',
            border=0,
            relief=tk.FLAT
        )
        self.command_entry.pack(fill=tk.X, pady=(5, 0), ipady=5)
        self.command_entry.bind('<Return>', self.execute_manual_command)
        
        self.log_message("ALPHA GUI with Gemini 2.5 Flash integration initialized successfully.")
        self.log_message("Click 'START ALPHA' to begin voice recognition.")
        self.log_message("Wake words: 'Alpha', 'Hey Alpha', 'OK Alpha'")
        self.log_message("Enhanced WhatsApp messaging with custom messages!")
        self.log_message(f"AI Mode: {'ENABLED' if self.ai_enabled else 'DISABLED'} - Ask me anything!")
        self.test_ai_connection()

        
    def get_personalized_greeting(self):
        """Generate a personalized greeting based on time and user"""
        current_hour = datetime.datetime.now().hour
        
        if current_hour < 12:
            time_greeting = "Good morning"
        elif current_hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        if self.user_name:
            return f"{time_greeting}, {self.user_name}! " + random.choice([
                "How are you feeling today?",
                "What can I help you with today?",
                "Ready to make your day amazing?",
                "Hope you're having a wonderful day!",
                "I'm here to assist you with anything you need!"
            ])
        else:
            return f"{time_greeting}! " + random.choice(self.greetings)
    
    def ask_user_name(self):
        """Ask for user's name on first run"""
        self.talk("Hello! I'm ALPHA, your personal AI assistant. What should I call you?")
        self.log_message("Asking for user's name...")
        
        try:
            name = self.listen(timeout=10)
            if name:
                name = name.replace("my name is", "").replace("call me", "").replace("i'm", "").replace("i am", "").strip()
                if name and len(name) > 0:
                    self.user_name = name.title()
                    self.talk(f"Nice to meet you, {self.user_name}! I'm excited to be your assistant.")
                    self.log_message(f"User name set to: {self.user_name}")
                    return True
            
            self.talk("That's okay! I'll just call you friend for now. You can tell me your name anytime by saying 'my name is' followed by your name.")
            self.user_name = "Friend"
            return True
            
        except Exception as e:
            self.log_message(f"Error getting user name: {str(e)}")
            self.user_name = "Friend"
            self.talk("I'll call you friend for now!")
            return True
    
    def test_ai_connection(self):
        """Test Gemini API connection"""
        if not self.ai_enabled:
            self.ai_status_label.config(text="🤖 AI: DISABLED", fg='#888888')
            self.log_message("AI disabled due to initialization error")
            return
        
        try:
            response = self.gemini_model.generate_content("Hi", generation_config={"max_output_tokens": 10})
            if response.text.strip():
                self.ai_status_label.config(text="🤖 AI: READY", fg='#00ffaa')
                self.log_message("Gemini 2.5 Flash connection successful!")
            else:
                self.ai_status_label.config(text="🤖 AI: ERROR", fg='#ff4444')
                self.log_message("Gemini 2.5 Flash connection failed - will use basic mode")
                self.ai_enabled = False
        except Exception as e:
            self.ai_status_label.config(text="🤖 AI: ERROR", fg='#ff4444')
            self.log_message(f"Gemini 2.5 Flash initialization error: {str(e)}")
            self.ai_enabled = False
    
    def call_gemini_api(self, message, test_mode=False):
        """Call Gemini API using SDK"""
        if not self.ai_enabled:
            self.log_message("AI is disabled")
            return None
        
        try:
            system_prompt = f"You are ALPHA, a helpful AI assistant for {self.user_name if self.user_name else 'the user'}. Keep responses concise and natural for voice interaction. You can help with general questions, provide information, have conversations, and assist with various tasks. If asked to do something that requires system access (like opening files or controlling software), politely explain that you can provide information but the user should use voice commands for system actions. Be friendly, helpful, and personable."
            
            # Prepare chat history
            history = []
            if not test_mode and self.conversation_history:
                for msg in self.conversation_history[-6:]:
                    history.append({"role": msg["role"], "parts": [msg["content"]]})
            
            # Generate content
            prompt = message if not test_mode else "Hi"
            response = self.gemini_model.generate_content(
                [system_prompt, *history, {"role": "user", "parts": [prompt]}],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 150 if not test_mode else 10
                }
            )
            
            if response.text.strip():
                ai_response = response.text.strip()
                if not test_mode:
                    self.conversation_history.append({"role": "user", "content": message})
                    self.conversation_history.append({"role": "assistant", "content": ai_response})
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]  
                return ai_response
            else:
                self.log_message("Gemini API returned empty response")
                return None
                
        except Exception as e:
            self.log_message(f"Gemini API error: {str(e)}")
            return None
    
    def is_basic_command(self, command):
        """Check if command should be handled by basic functions"""
        command_lower = command.lower()
        
        if self.waiting_for_message:
            return True
        
        return any(keyword in command_lower for keyword in self.basic_commands)
    
    def toggle_ai(self):
        """Toggle AI functionality"""
        self.ai_enabled = not self.ai_enabled
        if self.ai_enabled:
            self.ai_status_label.config(text="🤖 AI: READY", fg='#00ffaa')
            self.talk("AI mode enabled. I can now help with general questions and conversations.")
            self.log_message("AI mode enabled")
        else:
            self.ai_status_label.config(text="🤖 AI: OFF", fg='#888888')
            self.talk("AI mode disabled. Basic commands only.")
            self.log_message("AI mode disabled")
    
    def log_message(self, message):
        """Add message to console log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.console.insert(tk.END, log_entry)
        self.console.see(tk.END)
        self.root.update_idletasks()
        
    def talk(self, text):
        """Text to speech with GUI logging"""
        self.log_message(f"ALPHA: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
    def listen(self, timeout=5):
        """Listen for voice commands with improved error handling"""
        try:
            with sr.Microphone() as source:
                self.listening_label.config(text="🎤 LISTENING...")
                if self.wake_word_mode and not self.waiting_for_message:
                    self.log_message("Listening for wake word...")
                elif self.waiting_for_message:
                    self.log_message(f"Listening for message content for {self.current_contact}...")
                else:
                    self.log_message("Listening for command...")
                
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
                
            self.listening_label.config(text="🔄 PROCESSING...")
            command = self.recognizer.recognize_google(audio)
            self.log_message(f"USER: {command}")
            self.listening_label.config(text="")
            return command.lower()
            
        except sr.RequestError as e:
            self.log_message(f"Speech recognition service error: {str(e)}")
            self.talk("Sorry, I'm having trouble connecting to the speech service. Please check your internet connection.")
            self.listening_label.config(text="")
            return ""
        except sr.WaitTimeoutError:
            self.listening_label.config(text="")
            return ""
        except sr.UnknownValueError:
            if not self.wake_word_mode and not self.waiting_for_message:
                self.talk("Sorry, I didn't catch that.")
            elif self.waiting_for_message:
                self.talk("I didn't catch your message. Please try again or say 'cancel' to stop.")
            self.listening_label.config(text="")
            return ""
        except Exception as e:
            self.log_message(f"Error in speech recognition: {str(e)}")
            self.talk("Sorry, there was an error with speech recognition.")
            self.listening_label.config(text="")
            return ""
        finally:
            # Restore original: no waveform to stop
            pass
    
    def check_wake_word(self, command):
        """Check if wake word is detected"""
        for wake_word in self.wake_words:
            if wake_word in command:
                return True
        return False
    
    def send_whatsapp_web_message(self, contact_name, message):
        """Send WhatsApp message via WhatsApp Web"""
        try:
            webbrowser.open("https://web.whatsapp.com")
            self.talk(f"Opening WhatsApp Web. Please scan the QR code if needed, then manually send '{message}' to {contact_name}")
            
            self.send_whatsapp_notification(contact_name, message)
            return True
            
        except Exception as e:
            self.log_message(f"WhatsApp Web error: {str(e)}")
            self.talk("Sorry, I couldn't open WhatsApp Web.")
            return False

    def send_whatsapp_notification(self, contact_name, message):
        """Send a Windows notification as fallback"""
        try:
            import win10toast
            toaster = win10toast.ToastNotifier()
            toaster.show_toast(
                "ALPHA WhatsApp Reminder",
                f"Remember to send '{message}' to {contact_name} on WhatsApp",
                duration=10
            )
            self.talk(f"I've created a reminder to send '{message}' to {contact_name}")
        except ImportError:
            self.log_message("win10toast not installed. Install with: pip install win10toast")
            self.talk(f"Please remember to send '{message}' to {contact_name} on WhatsApp")
        except Exception as e:
            self.log_message(f"Notification error: {str(e)}")
            self.talk(f"Please remember to send '{message}' to {contact_name} on WhatsApp")
    
    def get_installed_software_from_registry(self):
        """Get installed software from Windows registry"""
        software_list = []
        registry_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
        ]
        
        for reg_path in registry_paths:
            try:
                registry_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                for i in range(winreg.QueryInfoKey(registry_key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(registry_key, i)
                        subkey = winreg.OpenKey(registry_key, subkey_name)
                        try:
                            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            try:
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                software_list.append((display_name, install_location))
                            except FileNotFoundError:
                                pass
                        except FileNotFoundError:
                            pass
                        winreg.CloseKey(subkey)
                    except:
                        continue
                winreg.CloseKey(registry_key)
            except:
                continue
        
        return software_list
    
    def search_software_path(self, app_name):
        """Enhanced software path search"""
        self.talk(f"Searching for {app_name} on your system. Please wait...")
        
        app_lower = app_name.lower()
        for name, paths in self.software_paths.items():
            if name in app_lower or app_lower in name:
                for path in paths:
                    if os.path.exists(path):
                        self.talk(f"Found {app_name}, launching now.")
                        return path
        
        possible_dirs = [
            r"C:\Program Files", 
            r"C:\Program Files (x86)",
            os.path.expanduser(r"~\AppData\Local"),
            os.path.expanduser(r"~\AppData\Roaming")
        ]
        
        exe_variations = [
            app_name.lower().replace(" ", "") + ".exe",
            app_name.lower() + ".exe",
            app_name.replace(" ", "") + ".exe",
            app_name + ".exe"
        ]
        
        for base_dir in possible_dirs:
            if not os.path.exists(base_dir):
                continue
                
            try:
                for root, dirs, files in os.walk(base_dir):
                    level = root.replace(base_dir, '').count(os.sep)
                    if level >= 3:
                        dirs[:] = []
                        continue
                        
                    for file in files:
                        if file.lower().endswith('.exe'):
                            for exe_var in exe_variations:
                                if file.lower() == exe_var or app_name.lower() in file.lower():
                                    found_path = os.path.join(root, file)
                                    self.talk(f"Found {app_name}, launching now.")
                                    return found_path
            except (PermissionError, OSError):
                continue
        
        try:
            software_list = self.get_installed_software_from_registry()
            for display_name, install_location in software_list:
                if app_name.lower() in display_name.lower():
                    if install_location and os.path.exists(install_location):
                        for file in os.listdir(install_location):
                            if file.lower().endswith('.exe'):
                                exe_path = os.path.join(install_location, file)
                                self.talk(f"Found {app_name}, launching now.")
                                return exe_path
        except:
            pass
        
        return None

    def open_software(self, software_name):
        """Enhanced software opening"""
        software_name = software_name.strip()
        
        if "whatsapp" in software_name.lower():
            return self.open_whatsapp()
        
        for name, paths in self.software_paths.items():
            if name in software_name.lower() or software_name.lower() in name:
                for path in paths:
                    if os.path.exists(path):
                        try:
                            subprocess.Popen(path)
                            self.talk(f"Opening {name}")
                            return
                        except Exception as e:
                            self.log_message(f"Failed to open {path}: {str(e)}")
                            continue
        
        try:
            os.startfile(software_name)
            self.talk(f"Opening {software_name}")
            return
        except:
            pass
        
        fallback_path = self.search_software_path(software_name)
        if fallback_path:
            try:
                subprocess.Popen(fallback_path)
            except Exception as e:
                self.talk(f"Found {software_name} but couldn't launch it.")
                self.log_message(f"Launch error: {str(e)}")
        else:
            self.talk(f"Could not locate {software_name} on your system.")

    def close_software(self, software_name):
        """Enhanced software closing"""
        software_name = software_name.strip().lower()
        closed_count = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                proc_name = proc.info['name'].lower()
                proc_exe = proc.info.get('exe', '').lower() if proc.info.get('exe') else ''
                
                if (software_name in proc_name or 
                    proc_name.startswith(software_name) or
                    software_name in proc_exe):
                    
                    try:
                        process = psutil.Process(proc.info['pid'])
                        process.terminate()
                        process.wait(timeout=3)
                        closed_count += 1
                        self.log_message(f"Terminated process: {proc.info['name']} (PID: {proc.info['pid']})")
                    except psutil.TimeoutExpired:
                        try:
                            process.kill()
                            closed_count += 1
                            self.log_message(f"Force killed process: {proc.info['name']} (PID: {proc.info['pid']})")
                        except:
                            pass
                    except Exception as e:
                        self.log_message(f"Failed to close {proc.info['name']}: {str(e)}")
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        if closed_count > 0:
            self.talk(f"Closed {closed_count} instance(s) of {software_name}")
        else:
            self.talk(f"{software_name} is not running or could not be closed.")

    def search_files(self, filename, file_extensions=None):
        """Search for files in common directories"""
        if file_extensions is None:
            file_extensions = ['*']
        
        found_files = []
        search_term = filename.lower()
        
        self.talk(f"Searching for {filename}...")
        
        for directory in self.search_directories:
            if not os.path.exists(directory):
                continue
                
            try:
                for extension in file_extensions:
                    pattern = f"*{search_term}*{extension}"
                    files = glob.glob(os.path.join(directory, pattern), recursive=False)
                    
                    try:
                        for root, dirs, files_in_dir in os.walk(directory):
                            level = root.replace(directory, '').count(os.sep)
                            if level >= 2:
                                dirs[:] = []
                                continue
                                
                            for file in files_in_dir:
                                if search_term in file.lower():
                                    found_files.append(os.path.join(root, file))
                    except (PermissionError, OSError):
                        continue
                        
            except (PermissionError, OSError):
                continue
        
        return found_files[:10]

    def open_file(self, filename):
        """Open a file by searching for it"""
        found_files = self.search_files(filename)
        
        if found_files:
            file_to_open = found_files[0]
            try:
                os.startfile(file_to_open)
                self.talk(f"Opening {os.path.basename(file_to_open)}")
                self.log_message(f"Opened file: {file_to_open}")
            except Exception as e:
                self.talk("Sorry, I couldn't open that file.")
                self.log_message(f"File open error: {str(e)}")
        else:
            self.talk(f"Could not find any file named {filename}")

    def play_on_youtube(self, command):
        """Play song on YouTube"""
        if "play" in command and "on youtube" in command:
            try:
                song = command.replace("play", "").replace("on youtube", "").strip()
                self.talk(f"Playing {song} on YouTube.")
                pywhatkit.playonyt(song)
            except Exception as e:
                self.talk("Sorry, I couldn't play that on YouTube.")
                self.log_message(f"YouTube error: {str(e)}")

    def play_on_spotify(self, command):
        """Play song on Spotify"""
        if "play" in command and "on spotify" in command:
            try:
                song = command.replace("play", "").replace("on spotify", "").strip()
                self.talk(f"Opening Spotify and searching {song}")
                self.open_software("spotify")
                webbrowser.open(f"https://open.spotify.com/search/{song.replace(' ', '%20')}")
            except Exception as e:
                self.talk("Sorry, I couldn't play that on Spotify.")
                self.log_message(f"Spotify error: {str(e)}")
    
    def process_command(self, command):
        """Enhanced command processing"""
        if not command:
            return
        
        if self.waiting_for_message:
            self.log_message(f"In message composition mode. Processing: {command}")
            self.handle_message_composition(command)
            return
            
        self.log_message(f"Processing command: {command}")
        
        if self.is_basic_command(command):
            self.process_basic_command(command)
        elif self.ai_enabled:
            self.process_ai_command(command)
        else:
            self.talk("I didn't understand that command. Try saying 'open [software name]', 'close [software name]', 'message [contact name]', 'search file [filename]', or 'open file [filename]'")
    
    def process_ai_command(self, command):
        """Process command using Gemini"""
        try:
            self.ai_status_label.config(text="🤖 AI: THINKING...", fg='#ffaa00')
            self.log_message("Consulting Gemini 2.5 Flash for response...")
            
            ai_response = self.call_gemini_api(command)
            
            if ai_response:
                self.ai_status_label.config(text="🤖 AI: READY", fg='#00ffaa')
                self.talk(ai_response)
            else:
                self.ai_status_label.config(text="🤖 AI: ERROR", fg='#ff4444')
                self.talk("Sorry, I'm having trouble connecting to my AI brain right now. Let me try to help with basic commands instead.")
                self.process_basic_command(command)
                
        except Exception as e:
            self.log_message(f"AI processing error: {str(e)}")
            self.ai_status_label.config(text="🤖 AI: ERROR", fg='#ff4444')
            self.talk("Sorry, I encountered an error with AI processing. Using basic mode.")
            self.process_basic_command(command)
        finally:
            pass
    
    def get_whatsapp_path(self):
        """Find WhatsApp installation path"""
        possible_paths = [
            os.path.expanduser(r"~\AppData\Local\WhatsApp\WhatsApp.exe"),
            os.path.expanduser(r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\WhatsApp\WhatsApp.lnk"),
            r"C:\Users\{}\AppData\Local\WhatsApp\WhatsApp.exe".format(os.getlogin()),
            r"C:\Program Files\WhatsApp\WhatsApp.exe",
            r"C:\Program Files (x86)\WhatsApp\WhatsApp.exe",
            os.path.expanduser(r"~\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\WhatsApp.exe"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        search_dirs = [
            os.path.expanduser("~\\AppData\\Local"),
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            os.path.expanduser("~\\AppData\\Roaming")
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                try:
                    for root, dirs, files in os.walk(search_dir):
                        if "whatsapp" in root.lower():
                            for file in files:
                                if file.lower() == "whatsapp.exe":
                                    return os.path.join(root, file)
                except (PermissionError, OSError):
                    continue
        
        return None

    def open_whatsapp(self):
        """Enhanced WhatsApp opening"""
        try:
            whatsapp_path = self.get_whatsapp_path()
            if whatsapp_path:
                try:
                    subprocess.Popen(whatsapp_path, shell=True)
                    self.talk("Opening WhatsApp")
                    return True
                except Exception as e:
                    self.log_message(f"Failed to open WhatsApp directly: {str(e)}")
            
            try:
                os.system("start whatsapp:")
                self.talk("Opening WhatsApp")
                return True
            except:
                pass
            
            try:
                webbrowser.open("https://web.whatsapp.com")
                self.talk("Opening WhatsApp Web in your browser")
                return True
            except:
                pass
            
            try:
                subprocess.run(["cmd", "/c", "start", "whatsapp:"], shell=True)
                self.talk("Opening WhatsApp")
                return True
            except:
                pass
                
            self.talk("Could not find WhatsApp. Please make sure it's installed.")
            return False
            
        except Exception as e:
            self.log_message(f"WhatsApp opening error: {str(e)}")
            self.talk("Sorry, I couldn't open WhatsApp.")
            return False

    def send_whatsapp_message_pywhatkit(self, contact_name, message):
        """Enhanced WhatsApp message sending"""
        try:
            contact_lower = contact_name.lower()
            if contact_lower in self.whatsapp_contacts:
                phone_number = self.whatsapp_contacts[contact_lower]
                
                now = datetime.datetime.now()
                send_time = now + datetime.timedelta(minutes=2)
                
                self.talk(f"Scheduling message to {contact_name} for {send_time.strftime('%H:%M')}")
                self.log_message(f"Sending message: '{message}' to {contact_name} ({phone_number}) at {send_time.strftime('%H:%M')}")
                
                try:
                    pywhatkit.sendwhatmsg(phone_number, message, send_time.hour, send_time.minute)
                    self.log_message("Message scheduled successfully via pywhatkit")
                    return True
                except Exception as pywhatkit_error:
                    self.log_message(f"Pywhatkit error: {str(pywhatkit_error)}")
                    self.talk("There was an issue with automatic sending. Opening WhatsApp Web instead.")
                    return self.send_whatsapp_web_message(contact_name, message)
                    
            else:
                self.talk(f"I don't have {contact_name}'s phone number in my contacts.")
                self.log_message(f"Available contacts: {', '.join(self.whatsapp_contacts.keys())}")
                return False
                
        except Exception as e:
            self.log_message(f"WhatsApp message error: {str(e)}")
            self.talk("Sorry, I couldn't send the WhatsApp message.")
            return False

    def initiate_message_composition(self, contact_name, message_type="custom"):
        """Start composing a custom message"""
        contact_lower = contact_name.lower()
        
        if contact_lower not in self.whatsapp_contacts:
            self.talk(f"I don't have {contact_name}'s contact information. Available contacts are: {', '.join(self.whatsapp_contacts.keys())}")
            return False
        
        self.waiting_for_message = True
        self.current_contact = contact_name
        self.message_type = message_type
        self.message_status_label.config(text=f"📝 COMPOSING MESSAGE TO {contact_name.upper()}")
        
        self.talk(f"What message would you like to send to {contact_name}? Please speak clearly and tell me your message.")
        
        return True

    def handle_message_composition(self, message_content):
        """Handle message content in composition mode"""
        if message_content.lower() in ['cancel', 'stop', 'never mind', 'abort']:
            self.waiting_for_message = False
            self.current_contact = ""
            self.message_type = ""
            self.message_status_label.config(text="")
            self.talk("Message cancelled.")
            return True
        
        message_content = message_content.strip()
        
        if not message_content:
            self.talk("I didn't get your message. Please try again or say 'cancel' to stop.")
            return False
        
        self.talk(f"You want to send '{message_content}' to {self.current_contact}. Should I send this message? Say 'yes', 'okay', or 'send' to confirm, or 'no' to change it.")
        self.message_status_label.config(text=f"📝 CONFIRM MESSAGE TO {self.current_contact.upper()}")
        
        confirmation_attempts = 0
        max_attempts = 3
        
        while confirmation_attempts < max_attempts:
            try:
                self.log_message(f"Waiting for confirmation... (Attempt {confirmation_attempts + 1}/{max_attempts})")
                confirmation = self.listen(timeout=15)
                
                if confirmation:
                    confirmation_lower = confirmation.lower()
                    self.log_message(f"Received confirmation: '{confirmation}'")
                    
                    positive_words = ['okay', 'ok', 'yes', 'send', 'go ahead', 'confirm', 'sure', 'yep', 'yeah']
                    negative_words = ['no', 'nope', 'cancel', 'change', 'edit', 'modify', 'different']
                    
                    if any(word in confirmation_lower for word in positive_words):
                        self.log_message("Positive confirmation received - sending message")
                        success = self.send_whatsapp_message_pywhatkit(self.current_contact, message_content)
                        self.waiting_for_message = False
                        self.current_contact = ""
                        self.message_type = ""
                        self.message_status_label.config(text="")
                        
                        if success:
                            self.talk("Perfect! Your message has been scheduled to send.")
                        else:
                            self.talk("Sorry, there was an issue sending the message.")
                        return True
                        
                    elif any(word in confirmation_lower for word in negative_words):
                        self.log_message("Negative confirmation received - asking for new message")
                        self.talk(f"Okay, what message would you like to send to {self.current_contact} instead?")
                        self.message_status_label.config(text=f"📝 COMPOSING MESSAGE TO {self.current_contact.upper()}")
                        return False
                        
                    else:
                        confirmation_attempts += 1
                        if confirmation_attempts < max_attempts:
                            self.talk("I didn't understand your response. Please say 'yes' or 'okay' to send the message, or 'no' to change it.")
                        else:
                            self.talk("I'm having trouble understanding. Message cancelled.")
                            self.waiting_for_message = False
                            self.current_contact = ""
                            self.message_type = ""
                            self.message_status_label.config(text="")
                            return True
                else:
                    confirmation_attempts += 1
                    if confirmation_attempts < max_attempts:
                        self.talk("I didn't hear a response. Please say 'yes' to send the message or 'no' to change it.")
                    else:
                        self.talk("No response received. Message cancelled.")
                        self.waiting_for_message = False
                        self.current_contact = ""
                        self.message_type = ""
                        self.message_status_label.config(text="")
                        return True
                    
            except Exception as e:
                self.log_message(f"Error in message confirmation: {str(e)}")
                confirmation_attempts += 1
                if confirmation_attempts < max_attempts:
                    self.talk("Sorry, there was an error. Please try again - say 'yes' to send or 'no' to change the message.")
                else:
                    self.talk("Sorry, there was an error. Message cancelled.")
                    self.waiting_for_message = False
                    self.current_contact = ""
                    self.message_type = ""
                    self.message_status_label.config(text="")
                    return True
        
        return False

    def process_basic_command(self, command):
        """Process basic built-in commands"""
        if "time" in command:
            time_now = datetime.datetime.now().strftime("%I:%M %p")
            self.talk(f"The time is {time_now}")

        elif "date" in command:
            date_now = datetime.datetime.now().strftime("%B %d, %Y")
            self.talk(f"Today is {date_now}")

        elif "my name is" in command:
            name = command.replace("my name is", "").strip()
            if name:
                self.user_name = name.title()
                self.talk(f"Nice to meet you, {self.user_name}! I'll remember that.")
                self.log_message(f"User name updated to: {self.user_name}")
            else:
                self.talk("Please tell me your name after saying 'my name is'")

        elif "what is my name" in command or "who am i" in command:
            if self.user_name:
                self.talk(f"You are {self.user_name}!")
            else:
                self.talk("I don't know your name yet. Please tell me by saying 'my name is' followed by your name.")

        elif "open" in command and "file" in command:
            parts = command.split("file")
            if len(parts) > 1:
                filename = parts[1].strip()
                self.open_file(filename)
            else:
                self.talk("Please specify which file to open")

        elif "search" in command and "file" in command:
            parts = command.split("file")
            if len(parts) > 1:
                filename = parts[1].strip()
                found_files = self.search_files(filename)
                if found_files:
                    self.talk(f"Found {len(found_files)} files matching {filename}")
                    for i, file in enumerate(found_files[:5], 1):
                        self.log_message(f"{i}. {file}")
                else:
                    self.talk(f"No files found matching {filename}")

        elif ("message" in command or "send message" in command) and any(contact in command for contact in self.whatsapp_contacts.keys()):
            contact_found = None
            for contact in self.whatsapp_contacts.keys():
                if contact in command:
                    contact_found = contact
                    break
            
            if contact_found:
                self.initiate_message_composition(contact_found, "custom")
            else:
                self.talk("I couldn't identify the contact. Please say 'message' followed by the contact name.")

        elif "whatsapp" in command:
            if any(contact in command for contact in self.whatsapp_contacts.keys()):
                contact_found = None
                for contact in self.whatsapp_contacts.keys():
                    if contact in command:
                        contact_found = contact
                        break
                
                if contact_found:
                    self.initiate_message_composition(contact_found, "custom")
            else:
                self.open_whatsapp()

        elif "open" in command:
            if "settings" in command:
                self.talk("Opening Windows settings.")
                os.system("start ms-settings:")
            elif "ALPHA dot p y" in command or "ALPHA.py" in command:
                self.talk("Opening ALPHA Python file.")
                subprocess.Popen(["notepad", "ALPHA.py"])
            else:
                software_name = command.split("open")[-1].strip()
                if software_name:
                    self.open_software(software_name)
                else:
                    self.talk("Please specify what to open")

        elif "close" in command or "stop" in command:
            if "close" in command:
                app_to_close = command.split("close")[-1].strip()
            else:
                app_to_close = command.split("stop")[-1].strip()
            
            if app_to_close:
                self.close_software(app_to_close)
            else:
                self.talk("Please specify what to close")

        elif "search" in command and "youtube" in command:
            search_term = command.split("search")[-1].split("on youtube")[0].strip()
            self.talk(f"Searching {search_term} on YouTube")
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_term}")

        elif "search" in command and "google" in command:
            search_term = command.split("search")[-1].split("on google")[0].strip()
            self.talk(f"Searching {search_term} on Google")
            webbrowser.open(f"https://www.google.com/search?q={search_term}")

        elif "play" in command and "on youtube" in command:
            self.play_on_youtube(command)

        elif "play" in command and "on spotify" in command:
            self.play_on_spotify(command)

        elif "sleep" in command or "go to sleep" in command:
            self.talk("Going to sleep mode. Say Alpha, Hey Alpha, or OK Alpha to wake me up.")
            self.wake_word_mode = True
            self.wake_mode_label.config(text="WAKE MODE: ON", fg='#ffaa00')

        elif "disable wake mode" in command or "always listen" in command:
            self.talk("Disabling wake mode. I'll always listen for commands now.")
            self.wake_word_mode = False
            self.wake_mode_label.config(text="WAKE MODE: OFF", fg='#00ff00')

        elif "wake" in command:
            self.talk("I'm already awake!")

        elif "list contacts" in command or "show contacts" in command:
            contacts = ", ".join(self.whatsapp_contacts.keys())
            self.talk(f"My WhatsApp contacts are: {contacts}")
            self.log_message(f"WhatsApp contacts: {contacts}")

        elif "enable ai" in command or "turn on ai" in command:
            if not self.ai_enabled:
                self.toggle_ai()
            else:
                self.talk("AI is already enabled!")

        elif "disable ai" in command or "turn off ai" in command:
            if self.ai_enabled:
                self.toggle_ai()
            else:
                self.talk("AI is already disabled!")

        elif "clear conversation" in command or "clear history" in command:
            self.conversation_history = []
            self.talk("Conversation history cleared!")
            self.log_message("AI conversation history cleared")

        elif "how are you" in command:
            responses = [
                f"I'm doing great, {self.user_name if self.user_name else 'friend'}! Thanks for asking. How are you?",
                "I'm fantastic and ready to help! How can I assist you today?",
                "I'm excellent! What can I do for you?",
                "I'm wonderful, thank you for asking! How are you feeling today?"
            ]
            self.talk(random.choice(responses))

        elif "thank you" in command or "thanks" in command:
            responses = [
                f"You're welcome, {self.user_name if self.user_name else 'friend'}!",
                "Happy to help! That's what I'm here for!",
                "My pleasure! Anything else I can do for you?",
                "You're welcome! I'm always here when you need me!"
            ]
            self.talk(random.choice(responses))

        elif "good morning" in command:
            responses = [
                f"Good morning, {self.user_name if self.user_name else 'friend'}! Hope you have a wonderful day ahead!",
                "Good morning! Ready to make today amazing?",
                "Morning! What can I help you with today?"
            ]
            self.talk(random.choice(responses))

        elif "good afternoon" in command:
            responses = [
                f"Good afternoon, {self.user_name if self.user_name else 'friend'}! How's your day going?",
                "Good afternoon! Hope you're having a productive day!",
                "Afternoon! What can I assist you with?"
            ]
            self.talk(random.choice(responses))

        elif "good evening" in command or "good night" in command:
            responses = [
                f"Good evening, {self.user_name if self.user_name else 'friend'}! Hope you had a great day!",
                "Good evening! How can I help you wind down?",
                "Evening! What would you like to do?"
            ]
            self.talk(random.choice(responses))

        elif "i love you" in command or "love you" in command:
            responses = [
                f"Aww, that's so sweet, {self.user_name if self.user_name else 'friend'}! I care about you too!",
                "You're amazing! I'm here whenever you need me!",
                "That means a lot to me! I'm happy to be your assistant!"
            ]
            self.talk(random.choice(responses))

        elif "tell me a joke" in command or "joke" in command:
            jokes = [
                "Why don't scientists trust atoms? Because they might be up to something!",
                "I told my computer a joke about UDP, but it didn't get it.",
                "Why did the programmer quit their job? Because they didn't get arrays!",
                "How do you comfort a JavaScript bug? You console it!",
                "Why do Python programmers prefer snake_case? Because they can't C# clearly!"
            ]
            self.talk(random.choice(jokes))

        elif "what can you do" in command or "help me" in command or "what are your features" in command:
            help_text = f"Hi {self.user_name if self.user_name else 'friend'}! I can help you with many things: " + \
                        "I can open and close software, send custom WhatsApp messages, search and open files, " + \
                        "play music on YouTube or Spotify, search Google, tell you the time and date, " + \
                        "have conversations with AI, and much more! Just ask me naturally, like 'message John' " + \
                        "or 'open Chrome' or 'what's the weather like?'"
            self.talk(help_text)

        elif "weather" in command:
            if self.ai_enabled:
                self.process_ai_command(command)
            else:
                self.talk("For weather information, I need AI mode enabled. You can enable it by saying 'enable AI' or by clicking the toggle AI button.")

        elif "exit" in command or "bye" in command or "shutdown" in command:
            farewell_messages = [
                f"Goodbye, {self.user_name if self.user_name else 'friend'}! Have a wonderful day!",
                "See you later! Take care!",
                "Bye! It was great helping you today!",
                "Farewell! Looking forward to our next conversation!"
            ]
            self.talk(random.choice(farewell_messages))
            self.stop_ALPHA()

        else:
            if self.ai_enabled:
                self.process_ai_command(command)
            else:
                helpful_suggestions = [
                    "I didn't understand that command. You can try:",
                    "• 'open [software name]' to launch applications",
                    "• 'message [contact name]' to send WhatsApp messages", 
                    "• 'search file [filename]' to find files",
                    "• 'what time is it' for current time",
                    "• 'tell me a joke' for entertainment",
                    "• 'what can you do' to see all features",
                    "• Enable AI mode for general conversations!"
                ]
                self.talk(" ".join(helpful_suggestions))
    
    def toggle_wake_mode(self):
        """Toggle wake word mode"""
        self.wake_word_mode = not self.wake_word_mode
        if self.wake_word_mode:
            self.wake_mode_label.config(text="WAKE MODE: ON", fg='#ffaa00')
            self.talk("Wake mode enabled. Say Alpha, Hey Alpha, or OK Alpha to activate.")
        else:
            self.wake_mode_label.config(text="WAKE MODE: OFF", fg='#00ff00')
            self.talk("Wake mode disabled. Always listening for commands.")
    
    def ALPHA_loop(self):
        """ALPHA listening loop"""
        if self.first_run:
            self.first_run = False
            if not self.user_name:
                self.ask_user_name()
            
            greeting = self.get_personalized_greeting()
            self.talk(greeting)
            self.talk("I'm online and ready to help! You can ask me anything or give me commands.")
        else:
            greeting = f"ALPHA back online and ready to assist{', ' + self.user_name if self.user_name else ''}!"
            self.talk(greeting)
        
        while self.is_running:
            if self.is_listening:
                try:
                    if self.wake_word_mode and not self.waiting_for_message:
                        command = self.listen(timeout=5)
                        if command and self.check_wake_word(command):
                            responses = [
                                "Yes, I'm listening.",
                                "How can I help you?",
                                "What can I do for you?",
                                "I'm here, what do you need?",
                                f"Yes {self.user_name if self.user_name else 'friend'}?"
                            ]
                            self.talk(random.choice(responses))
                            actual_command = self.listen(timeout=10)
                            if actual_command and self.is_running:
                                self.process_command(actual_command)
                    else:
                        command = self.listen(timeout=5)
                        if command and self.is_running:
                            self.process_command(command)
                except Exception as e:
                    self.log_message(f"Error in ALPHA loop: {str(e)}")
                    time.sleep(1)
            else:
                time.sleep(0.1)
    
    def toggle_ALPHA(self):
        """Start or stop ALPHA"""
        if not self.is_running:
            self.start_ALPHA()
        else:
            self.stop_ALPHA()
    
    def start_ALPHA(self):
        """Start ALPHA voice recognition"""
        self.is_running = True
        self.is_listening = True
        self.status_label.config(text="● ONLINE", fg='#00ff00')
        self.start_button.config(text="🟥 STOP ALPHA", bg='#444444', fg='#ff0000')
        self.manual_listen_button.config(state=tk.NORMAL)
        
        self.log_message("ALPHA started successfully!")
        
        self.ALPHA_thread = threading.Thread(target=self.ALPHA_loop, daemon=True)
        self.ALPHA_thread.start()
    
    def stop_ALPHA(self):
        """Stop ALPHA voice recognition"""
        self.is_running = False
        self.is_listening = False
        self.waiting_for_message = False
        self.current_contact = ""
        self.message_type = ""
        self.status_label.config(text="● OFFLINE", fg='#ff4444')
        self.start_button.config(text="🔴 START ALPHA", bg='#004400', fg='#00ff00')
        self.manual_listen_button.config(state=tk.DISABLED)
        self.listening_label.config(text="")
        self.message_status_label.config(text="")
        
        self.log_message("ALPHA stopped.")
    
    def manual_listen(self):
        """Manual single listen command"""
        if self.is_running:
            self.is_listening = False
            command = self.listen()
            if command:
                self.process_command(command)
            self.is_listening = True
    
    def execute_manual_command(self, event=None):
        """Execute command from text input"""
        command = self.command_entry.get().strip()
        if command:
            self.command_entry.delete(0, tk.END)
            self.log_message(f"MANUAL COMMAND: {command}")
            self.process_command(command.lower())
            
    def clear_log(self):
        """Clear console log"""
        self.console.delete(1.0, tk.END)
        self.log_message("Console cleared.")
        
    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
            
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_ALPHA()
            
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    try:
        app = ALPHAGUI()
        app.run()
    except Exception as e:
        print(f"Error starting ALPHA GUI: {e}")
        input("Press Enter to exit...")

