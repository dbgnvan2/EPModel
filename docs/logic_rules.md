# EPModel: Emotional Process Logic Rules

### 1. SYSTEMIC TRIANGULATION (Core Logic)
- **The Rule of Three:** No relationship exists in isolation. All interactions must be modeled as potential triangles.
- **Reactivity Trigger:** When "Anxiety" in a dyad exceeds a threshold (default: 0.7), the model must search for a third "Node" to triangulate.
- **Node States:** Nodes (Individuals) have two primary variables: `Level of Differentiation` (Static) and `Chronic Anxiety` (Dynamic).

### 2. ANXIETY FLOW (Data Isolation)
- **Contagion Effect:** Anxiety is a shared systemic variable. An increase in Node A’s anxiety must proportionally impact linked Nodes B and C unless "Differentiation" acts as a buffer.
- **Negative Constraints:** Anxiety cannot be a negative value. Differentiation is a scale of 0.0 to 1.0.

### 3. FUNCTIONAL COLLAPSE (The Evaluator)
- **Thresholds:** If `Anxiety > Differentiation`, the Node enters "Reactive Mode" (Output: Binary 1).
- **Behavioral Outputs:** Reactive Mode triggers one of four patterns: 1. Conflict, 2. Distance, 3. Over/Under Functioning, 4. Triangulation.