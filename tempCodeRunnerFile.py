import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('DUMMY_DATA.csv')
    
    # Basic preprocessing
    # Fill missing values with median for numerical features
    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Convert categorical variables to numerical
    categorical_cols = ['Ethnicity', 'Gender']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes
    
    # Standardize numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data, numerical_cols, categorical_cols

# Create graph data from patient encounters
def create_graph_data(patient_data, numerical_cols, categorical_cols):
    # Use all features as node features
    node_features = patient_data[numerical_cols + categorical_cols].values
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Create a fully connected graph (each node connected to all others)
    num_nodes = len(patient_data)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    
    if len(edge_index) == 0:
        # Handle case with only one node
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=node_features, edge_index=edge_index)

# Siamese GNN Model
class SiameseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseGNN, self).__init__()
        self.gnn = nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward_one(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.gnn:
            x = F.relu(conv(x, edge_index))
        x = torch.mean(x, dim=0)  # Global mean pooling
        x = self.fc(x)
        return x
    
    def forward(self, data1, data2):
        out1 = self.forward_one(data1)
        out2 = self.forward_one(data2)
        return out1, out2

# Similarity function
def similarity_score(embedding1, embedding2):
    return F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

# Train the model
def train_model(data, numerical_cols, categorical_cols, epochs=50):
    # Create graph data for all patients
    graph_data = create_graph_data(data, numerical_cols, categorical_cols)
    
    # For simplicity, we'll just use the same graph data for both branches
    # In a real scenario, you'd have different patient graphs
    model = SiameseGNN(input_dim=len(numerical_cols + categorical_cols), hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy training with random pairs (in a real scenario, you'd have labeled pairs)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass through both branches
        out1, out2 = model(graph_data, graph_data)
        
        # Dummy loss (in a real scenario, use actual similarity labels)
        loss = F.mse_loss(out1, out2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            st.write(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return model

# Streamlit UI
def main():
    st.title("Patient Encounter Similarity with Siamese GNN")
    st.write("This app calculates similarity between patient encounters using a Graph Neural Network.")
    
    # Load data
    data, numerical_cols, categorical_cols = load_data()
    
    # Display raw data
    if st.checkbox("Show raw data"):
        st.write(data.head())
    
    # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model = train_model(data, numerical_cols, categorical_cols)
            st.success("Model trained successfully!")
            
            # Save model to session state
            st.session_state.model = model
            st.session_state.graph_data = create_graph_data(data, numerical_cols, categorical_cols)
    
    # Compare patients
    if 'model' in st.session_state:
        st.header("Compare Patient Encounters")
        
        # Select two patients to compare
        patient1 = st.selectbox("Select first patient:", data['encounter_id'].unique())
        patient2 = st.selectbox("Select second patient:", data['encounter_id'].unique())
        
        if st.button("Calculate Similarity"):
            # Get patient indices
            idx1 = data[data['encounter_id'] == patient1].index[0]
            idx2 = data[data['encounter_id'] == patient2].index[0]
            
            # Create subgraphs for each patient
            # For simplicity, we're using the same graph structure but different node features
            graph_data = st.session_state.graph_data
            patient_graph1 = Data(x=graph_data.x[idx1].unsqueeze(0), edge_index=torch.tensor([[], []], dtype=torch.long))
            patient_graph2 = Data(x=graph_data.x[idx2].unsqueeze(0), edge_index=torch.tensor([[], []], dtype=torch.long))
            
            # Get embeddings
            model = st.session_state.model
            model.eval()
            with torch.no_grad():
                emb1, emb2 = model(patient_graph1, patient_graph2)
            
            # Calculate similarity
            similarity = similarity_score(emb1, emb2)
            
            # Display results
            st.subheader("Results")
            st.write(f"Similarity score between patient {patient1} and {patient2}: {similarity:.4f}")
            
            # Visualize embeddings
            fig, ax = plt.subplots()
            ax.bar(['Similarity'], [similarity], color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Similarity Score')
            ax.set_title('Patient Encounter Similarity')
            st.pyplot(fig)
            
            # Show feature comparison
            st.subheader("Feature Comparison")
            patient1_data = data[data['encounter_id'] == patient1].iloc[0]
            patient2_data = data[data['encounter_id'] == patient2].iloc[0]
            
            compare_df = pd.DataFrame({
                'Feature': numerical_cols[:10] + categorical_cols,  # Show first 10 numerical for space
                f'Patient {patient1}': patient1_data[numerical_cols[:10] + categorical_cols].values,
                f'Patient {patient2}': patient2_data[numerical_cols[:10] + categorical_cols].values
            })
            st.dataframe(compare_df)
    
    # Explanation
    st.header("How It Works")
    st.write("""
    1. **Data Loading**: The patient encounter data is loaded and preprocessed.
    2. **Graph Construction**: Each patient encounter is treated as a node in a graph.
    3. **Siamese GNN**: Two identical GNN branches process patient data to generate embeddings.
    4. **Similarity Calculation**: Cosine similarity compares the embeddings to produce a score.
    5. **Visualization**: Results are displayed with similarity scores and feature comparisons.
    """)

if __name__ == "__main__":
    main()