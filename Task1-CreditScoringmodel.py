import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

def create_sample_data():
    """Generate realistic sample data for testing our credit scoring model"""
    np.random.seed(42)  # For reproducible results
    sample_size = 500
    
    # Create realistic financial profiles
    monthly_income = np.random.randint(2000, 20000, sample_size)
    total_debts = np.random.randint(0, 15000, sample_size)
    payment_history = np.random.choice(['good', 'average', 'poor'], sample_size)
    num_accounts = np.random.randint(1, 10, sample_size)
    loan_amount = np.random.randint(1000, 50000, sample_size)
    credit_utilization = np.round(np.random.uniform(0.1, 1.0, sample_size), 2)
    age = np.random.randint(18, 70, sample_size)
    credit_history_length = np.random.randint(1, 30, sample_size)
    previous_defaults = np.random.choice(['Yes', 'No'], sample_size)
    employment_status = np.random.choice(['Employed', 'Self-employed', 'Unemployed'], sample_size)
    
    # Create target variable (1 = creditworthy, 0 = not creditworthy)
    # Most people are creditworthy in our sample
    creditworthy = np.random.choice([0, 1], sample_size, p=[0.3, 0.7])
    
    # Put it all together in a nice dataframe
    data = pd.DataFrame({
        'Monthly Income': monthly_income,
        'Total Outstanding Debts': total_debts,
        'Payment History': payment_history,
        'Number of Credit Accounts': num_accounts,
        'Loan Amount': loan_amount,
        'Credit Utilization Ratio': credit_utilization,
        'Age': age,
        'Credit History Length': credit_history_length,
        'Previous Loan Defaults': previous_defaults,
        'Employment Status': employment_status,
        'Creditworthy': creditworthy
    })
    
    return data

def prepare_data_for_training(df):
    """Convert text data to numbers so our model can understand it"""
    # Set up encoders for categorical data
    payment_encoder = LabelEncoder()
    defaults_encoder = LabelEncoder()
    employment_encoder = LabelEncoder()
    
    # Convert text categories to numbers
    df['Payment History'] = payment_encoder.fit_transform(df['Payment History'])
    df['Previous Loan Defaults'] = defaults_encoder.fit_transform(df['Previous Loan Defaults'])
    df['Employment Status'] = employment_encoder.fit_transform(df['Employment Status'])
    
    # Separate features from target
    features = df.drop('Creditworthy', axis=1)
    target = df['Creditworthy']
    
    # Scale features so they're all on similar ranges
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target, scaler, payment_encoder, defaults_encoder, employment_encoder

def train_credit_model(X, y):
    """Train our logistic regression model"""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test how well our model performs
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print("How well our model performed:")
    print(classification_report(y_test, predictions))
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    
    # Test with a sample
    sample = X_test[0].reshape(1, -1)
    sample_prediction = model.predict(sample)[0]
    result = "Creditworthy" if sample_prediction == 1 else "Not Creditworthy"
    print(f"Sample test result: {result}")
    
    return model

def save_model_and_encoders(model, scaler, payment_enc, defaults_enc, employment_enc):
    """Save our trained model and all the encoders for later use"""
    model_package = {
        'model': model,
        'scaler': scaler,
        'le_payment': payment_enc,
        'le_defaults': defaults_enc,
        'le_employment': employment_enc
    }
    
    with open("credit_scoring_model.pkl", "wb") as file:
        pickle.dump(model_package, file)
    
    print("Model saved successfully!")

def load_saved_model():
    """Load our previously trained model"""
    try:
        with open("credit_scoring_model.pkl", "rb") as file:
            model_package = pickle.load(file)
        return model_package
    except FileNotFoundError:
        print("No saved model found. Please train the model first.")
        return None

class CreditScoringApp:
    """A nice GUI application for credit scoring"""
    
    def __init__(self):
        self.model_data = load_saved_model()
        if not self.model_data:
            # If no model exists, create and train one
            self.train_new_model()
        
        # Define feature names for analysis
        self.feature_names = [
            'Monthly Income', 'Total Outstanding Debts', 'Payment History',
            'Number of Credit Accounts', 'Loan Amount', 'Credit Utilization Ratio',
            'Age', 'Credit History Length', 'Previous Loan Defaults', 'Employment Status'
        ]
        
        self.setup_gui()
    
    def train_new_model(self):
        """Train a new model if none exists"""
        print("Training new model...")
        data = create_sample_data()
        X, y, scaler, pay_enc, def_enc, emp_enc = prepare_data_for_training(data)
        model = train_credit_model(X, y)
        save_model_and_encoders(model, scaler, pay_enc, def_enc, emp_enc)
        self.model_data = load_saved_model()
    
    def setup_gui(self):
        """Create the main application window"""
        self.root = tk.Tk()
        self.root.title("Credit Scoring Tool")
        self.root.geometry("550x700")
        self.root.configure(bg="#1a1a1a")
        
        self.create_header()
        self.create_input_form()
        self.create_buttons()
        self.create_results_section()
    
    def create_header(self):
        """Create the app header with title and tagline"""
        tagline = tk.Label(
            self.root,
            text="Smart Credit Decisions Made Simple",
            font=("Arial", 12, "italic"),
            bg="#1a1a1a",
            fg="#00cc66"
        )
        tagline.pack(pady=(20, 5))
        
        title = tk.Label(
            self.root,
            text="Credit Scoring Tool",
            font=("Arial", 24, "bold"),
            bg="#1a1a1a",
            fg="white"
        )
        title.pack(pady=(0, 10))
        
        # Add a nice divider line
        divider = tk.Frame(self.root, bg="#00cc66", height=3)
        divider.pack(fill="x", padx=50, pady=(0, 20))
    
    def create_input_form(self):
        """Create the form where users enter financial information"""
        self.form_frame = tk.Frame(self.root, bg="#2d2d2d", padx=20, pady=20)
        self.form_frame.pack(pady=10, padx=20, fill="both")
        
        # Store all our input fields
        self.inputs = {}
        
        # Income field
        self.add_input_field("Monthly Income ($):", "income", tk.Entry)
        
        # Debts field
        self.add_input_field("Total Outstanding Debts ($):", "debts", tk.Entry)
        
        # Payment history dropdown
        payment_combo = ttk.Combobox(self.form_frame, values=['good', 'average', 'poor'], state="readonly")
        payment_combo.current(0)
        self.add_input_field("Payment History:", "payment", None, payment_combo)
        
        # Number of accounts
        self.add_input_field("Number of Credit Accounts:", "accounts", tk.Entry)
        
        # Loan amount
        self.add_input_field("Requested Loan Amount ($):", "loan", tk.Entry)
        
        # Credit utilization
        self.add_input_field("Credit Utilization Ratio (0-1):", "utilization", tk.Entry)
        
        # Age
        self.add_input_field("Age:", "age", tk.Entry)
        
        # Credit history length
        self.add_input_field("Credit History Length (years):", "history", tk.Entry)
        
        # Previous defaults dropdown
        defaults_combo = ttk.Combobox(self.form_frame, values=['Yes', 'No'], state="readonly")
        defaults_combo.current(1)  # Default to 'No'
        self.add_input_field("Previous Loan Defaults:", "defaults", None, defaults_combo)
        
        # Employment status dropdown
        employment_combo = ttk.Combobox(self.form_frame, values=['Employed', 'Self-employed', 'Unemployed'], state="readonly")
        employment_combo.current(0)  # Default to 'Employed'
        self.add_input_field("Employment Status:", "employment", None, employment_combo)
    
    def add_input_field(self, label_text, field_name, widget_class, custom_widget=None):
        """Helper method to add labeled input fields to the form"""
        row = len(self.inputs)
        
        # Create label
        label = tk.Label(
            self.form_frame, 
            text=label_text,
            font=("Arial", 11),
            bg="#2d2d2d",
            fg="white"
        )
        label.grid(row=row, column=0, sticky="w", pady=8, padx=(0, 10))
        
        # Create input widget
        if custom_widget:
            widget = custom_widget
        else:
            widget = widget_class(self.form_frame, font=("Arial", 10))
        
        widget.grid(row=row, column=1, pady=8, sticky="ew")
        
        # Store reference to the widget
        self.inputs[field_name] = widget
        
        # Make the input column expand
        self.form_frame.grid_columnconfigure(1, weight=1)
    
    def create_buttons(self):
        """Create the main action buttons"""
        button_frame = tk.Frame(self.root, bg="#1a1a1a")
        button_frame.pack(pady=20)
        
        # Style the buttons
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Custom.TButton',
            background='#00cc66',
            foreground='white',
            font=("Arial", 12, "bold"),
            padding=10
        )
        style.map('Custom.TButton', background=[('active', '#00aa55')])
        
        # Predict button
        predict_btn = ttk.Button(
            button_frame,
            text="Check Credit Score",
            command=self.make_prediction,
            style='Custom.TButton'
        )
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Analysis button
        analysis_btn = ttk.Button(
            button_frame,
            text="Show Detailed Analysis",
            command=self.show_analysis,
            style='Custom.TButton'
        )
        analysis_btn.pack(side=tk.LEFT, padx=10)
    
    def create_results_section(self):
        """Create the area where results are displayed"""
        # Result label
        self.result_label = tk.Label(
            self.root,
            text="Enter information above and click 'Check Credit Score'",
            font=("Arial", 14, "bold"),
            bg="#1a1a1a",
            fg="#00cc66"
        )
        self.result_label.pack(pady=20)
        
        # Analysis text area
        self.analysis_text = tk.Text(
            self.root,
            height=10,
            width=60,
            font=("Arial", 10),
            bg="#2d2d2d",
            fg="white",
            wrap=tk.WORD
        )
        self.analysis_text.pack(pady=10, padx=20, fill="both", expand=True)
        self.analysis_text.config(state=tk.DISABLED)
    
    def get_user_input(self):
        """Collect and validate all user input from the form"""
        try:
            # Get values from all input fields
            income = float(self.inputs['income'].get())
            debts = float(self.inputs['debts'].get())
            payment_history = self.inputs['payment'].get()
            accounts = int(self.inputs['accounts'].get())
            loan_amount = float(self.inputs['loan'].get())
            utilization = float(self.inputs['utilization'].get())
            age = int(self.inputs['age'].get())
            history_length = int(self.inputs['history'].get())
            previous_defaults = self.inputs['defaults'].get()
            employment = self.inputs['employment'].get()
            
            # Encode categorical variables
            payment_encoded = self.model_data['le_payment'].transform([payment_history])[0]
            defaults_encoded = self.model_data['le_defaults'].transform([previous_defaults])[0]
            employment_encoded = self.model_data['le_employment'].transform([employment])[0]
            
            # Create feature array
            features = [
                income, debts, payment_encoded, accounts, loan_amount,
                utilization, age, history_length, defaults_encoded, employment_encoded
            ]
            
            return features, {
                'income': income,
                'debts': debts,
                'payment_history': payment_history,
                'accounts': accounts,
                'loan_amount': loan_amount,
                'utilization': utilization,
                'age': age,
                'history_length': history_length,
                'previous_defaults': previous_defaults,
                'employment': employment
            }
            
        except ValueError as e:
            raise ValueError("Please enter valid numbers in all numeric fields")
        except Exception as e:
            raise Exception(f"Error processing input: {str(e)}")
    
    def analyze_rejection_reasons(self, raw_data):
        """Analyze why the user was rejected and provide specific reasons"""
        reasons = []
        
        # Check debt-to-income ratio
        debt_to_income = (raw_data['debts'] / raw_data['income']) * 100 if raw_data['income'] > 0 else 100
        if debt_to_income > 40:
            reasons.append(f"High debt-to-income ratio ({debt_to_income:.1f}%). Ideal range is below 40%")
        
        # Check payment history
        if raw_data['payment_history'] == 'poor':
            reasons.append("Poor payment history indicates higher risk of default")
        elif raw_data['payment_history'] == 'average':
            reasons.append("Average payment history - improvement needed for better creditworthiness")
        
        # Check previous defaults
        if raw_data['previous_defaults'] == 'Yes':
            reasons.append("History of previous loan defaults significantly impacts creditworthiness")
        
        # Check employment status
        if raw_data['employment'] == 'Unemployed':
            reasons.append("Unemployment status indicates unstable income source")
        
        # Check credit utilization
        if raw_data['utilization'] > 0.8:
            reasons.append(f"High credit utilization ({raw_data['utilization']:.1%}). Keep below 80% for better scores")
        
        # Check credit history length
        if raw_data['history_length'] < 2:
            reasons.append("Limited credit history length makes it difficult to assess creditworthiness")
        
        # Check loan amount vs income ratio
        loan_to_income = (raw_data['loan_amount'] / (raw_data['income'] * 12)) * 100 if raw_data['income'] > 0 else 100
        if loan_to_income > 30:
            reasons.append(f"Requested loan amount is too high relative to annual income ({loan_to_income:.1f}% of annual income)")
        
        # Check age factor
        if raw_data['age'] < 21:
            reasons.append("Young age may indicate limited financial experience and stability")
        
        # Check income adequacy
        if raw_data['income'] < 3000:
            reasons.append("Low monthly income may not support loan repayment obligations")
        
        # Check number of accounts
        if raw_data['accounts'] > 8:
            reasons.append("High number of credit accounts may indicate over-extension")
        elif raw_data['accounts'] < 2:
            reasons.append("Limited number of credit accounts - insufficient credit diversity")
        
        return reasons
    
    def get_improvement_suggestions(self, reasons):
        """Provide improvement suggestions based on rejection reasons"""
        suggestions = []
        
        if any("debt-to-income" in reason for reason in reasons):
            suggestions.append("• Pay down existing debts to reduce debt-to-income ratio")
            suggestions.append("• Consider increasing income through additional employment")
        
        if any("payment history" in reason for reason in reasons):
            suggestions.append("• Make all future payments on time to improve payment history")
            suggestions.append("• Set up automatic payments to avoid missed payments")
        
        if any("previous loan defaults" in reason for reason in reasons):
            suggestions.append("• Wait 12-24 months after resolving defaults before reapplying")
            suggestions.append("• Consider a secured loan to rebuild credit reputation")
        
        if any("Unemployment" in reason for reason in reasons):
            suggestions.append("• Secure stable employment before applying for credit")
            suggestions.append("• Provide proof of alternative income sources if available")
        
        if any("credit utilization" in reason for reason in reasons):
            suggestions.append("• Pay down credit card balances to reduce utilization")
            suggestions.append("• Avoid closing old credit accounts to maintain available credit")
        
        if any("credit history" in reason for reason in reasons):
            suggestions.append("• Keep existing accounts open to build credit history length")
            suggestions.append("• Consider becoming an authorized user on a family member's account")
        
        if any("loan amount" in reason for reason in reasons):
            suggestions.append("• Request a smaller loan amount that better matches your income")
            suggestions.append("• Save for a larger down payment to reduce loan amount needed")
        
        if any("Low monthly income" in reason for reason in reasons):
            suggestions.append("• Increase income through additional work or skill development")
            suggestions.append("• Consider applying with a co-signer who has higher income")
        
        return suggestions
    
    def make_prediction(self):
        """Make a credit worthiness prediction based on user input"""
        try:
            features, raw_data = self.get_user_input()
            
            # Scale the features
            features_scaled = self.model_data['scaler'].transform([features])
            
            # Make prediction
            prediction = self.model_data['model'].predict(features_scaled)[0]
            probability = self.model_data['model'].predict_proba(features_scaled)[0][1]
            
            # Update result display
            if prediction == 1:
                result_text = f"✓ CREDITWORTHY (Confidence: {probability:.1%})"
                color = "#00cc66"
                
                # Show success message in analysis area
                analysis_text = "=== CREDIT APPROVED ===\n\n"
                analysis_text += "Congratulations! Your credit application shows positive indicators.\n\n"
                analysis_text += "Strong points in your application:\n"
                
                # Add positive factors
                debt_to_income = (raw_data['debts'] / raw_data['income']) * 100 if raw_data['income'] > 0 else 100
                if debt_to_income <= 40:
                    analysis_text += f"• Good debt-to-income ratio: {debt_to_income:.1f}%\n"
                if raw_data['payment_history'] == 'good':
                    analysis_text += "• Excellent payment history\n"
                if raw_data['previous_defaults'] == 'No':
                    analysis_text += "• No previous defaults\n"
                if raw_data['employment'] == 'Employed':
                    analysis_text += "• Stable employment status\n"
                
                analysis_text += "\nClick 'Show Detailed Analysis' for complete financial assessment."
                
            else:
                result_text = f"✗ NOT CREDITWORTHY (Confidence: {(1-probability):.1%})"
                color = "#ff4444"
                
                # Get rejection reasons and suggestions
                rejection_reasons = self.analyze_rejection_reasons(raw_data)
                suggestions = self.get_improvement_suggestions(rejection_reasons)
                
                # Show rejection analysis
                analysis_text = "=== CREDIT APPLICATION DECLINED ===\n\n"
                analysis_text += "We're sorry, but your application doesn't meet our current criteria.\n\n"
                
                if rejection_reasons:
                    analysis_text += "PRIMARY CONCERNS:\n"
                    for i, reason in enumerate(rejection_reasons, 1):
                        analysis_text += f"{i}. {reason}\n"
                    analysis_text += "\n"
                
                if suggestions:
                    analysis_text += "IMPROVEMENT RECOMMENDATIONS:\n"
                    for suggestion in suggestions:
                        analysis_text += f"{suggestion}\n"
                    analysis_text += "\n"
                
                analysis_text += "We encourage you to address these concerns and reapply in the future.\n"
                analysis_text += "Click 'Show Detailed Analysis' for complete financial assessment."
            
            self.result_label.config(text=result_text, fg=color)
            
            # Update analysis text area
            self.analysis_text.config(state=tk.NORMAL)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, analysis_text)
            self.analysis_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
    
    def show_analysis(self):
        """Display detailed analysis of the user's financial profile"""
        try:
            features, raw_data = self.get_user_input()
            features_scaled = self.model_data['scaler'].transform([features])
            probability = self.model_data['model'].predict_proba(features_scaled)[0][1]
            prediction = self.model_data['model'].predict(features_scaled)[0]
            
            # Create detailed analysis text
            analysis = "=== COMPREHENSIVE CREDIT ANALYSIS ===\n\n"
            analysis += f"Applicant Financial Profile:\n"
            analysis += f"• Monthly Income: ${raw_data['income']:,.2f}\n"
            analysis += f"• Total Outstanding Debts: ${raw_data['debts']:,.2f}\n"
            analysis += f"• Payment History: {raw_data['payment_history'].title()}\n"
            analysis += f"• Number of Credit Accounts: {raw_data['accounts']}\n"
            analysis += f"• Requested Loan Amount: ${raw_data['loan_amount']:,.2f}\n"
            analysis += f"• Credit Utilization Ratio: {raw_data['utilization']:.2f}\n"
            analysis += f"• Age: {raw_data['age']} years\n"
            analysis += f"• Credit History Length: {raw_data['history_length']} years\n"
            analysis += f"• Previous Defaults: {raw_data['previous_defaults']}\n"
            analysis += f"• Employment Status: {raw_data['employment']}\n\n"
            
            analysis += f"=== MODEL ASSESSMENT ===\n"
            analysis += f"Creditworthiness Probability: {probability:.1%}\n"
            
            if probability >= 0.7:
                analysis += "Risk Level: LOW - Strong candidate for approval\n"
            elif probability >= 0.5:
                analysis += "Risk Level: MODERATE - Consider with additional review\n"
            else:
                analysis += "Risk Level: HIGH - Recommend decline or additional requirements\n"
            
            # Calculate key financial ratios
            debt_to_income = (raw_data['debts'] / raw_data['income']) * 100 if raw_data['income'] > 0 else 100
            loan_to_income = (raw_data['loan_amount'] / (raw_data['income'] * 12)) * 100 if raw_data['income'] > 0 else 100
            
            analysis += f"\nKey Financial Ratios:\n"
            analysis += f"• Debt-to-Income Ratio: {debt_to_income:.1f}%\n"
            analysis += f"• Loan-to-Annual-Income Ratio: {loan_to_income:.1f}%\n"
            
            if debt_to_income < 20:
                analysis += "  → Excellent debt management\n"
            elif debt_to_income < 40:
                analysis += "  → Acceptable debt levels\n"
            else:
                analysis += "  → High debt burden - concerning\n"
            
            # If not creditworthy, show detailed reasons
            if prediction == 0:
                rejection_reasons = self.analyze_rejection_reasons(raw_data)
                suggestions = self.get_improvement_suggestions(rejection_reasons)
                
                analysis += "\n=== DETAILED REJECTION ANALYSIS ===\n"
                if rejection_reasons:
                    analysis += "Specific Issues Identified:\n"
                    for i, reason in enumerate(rejection_reasons, 1):
                        analysis += f"{i}. {reason}\n"
                
                if suggestions:
                    analysis += "\nRecommended Actions for Improvement:\n"
                    for suggestion in suggestions:
                        analysis += f"{suggestion}\n"
            
            # Display the analysis
            self.analysis_text.config(state=tk.NORMAL)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, analysis)
            self.analysis_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Starting Enhanced Credit Scoring Application...")
    
    # Check if we need to train a new model
    try:
        with open("credit_scoring_model.pkl", "rb") as f:
            print("Found existing model, loading...")
    except FileNotFoundError:
        print("No existing model found, will create one...")
    
    # Launch the application
    app = CreditScoringApp()
    app.run()
