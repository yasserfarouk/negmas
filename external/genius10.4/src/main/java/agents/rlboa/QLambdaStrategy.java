package agents.rlboa;

import java.util.ArrayList;
import java.util.Collections;

import genius.core.StrategyParameters;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;

public class QLambdaStrategy extends QlearningStrategy {

	double lambda;
	
	public QLambdaStrategy(NegotiationSession negotiationSession, OpponentModel opponentModel) {
		super(negotiationSession, opponentModel);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	protected void updateQFuction(AbstractState state, int action, double reward, AbstractState newState) {
		// initialize state if it is new

		// If agent hasn't done a opening bid, initialize action values to number of bins, otherwise
		// just 3 values (up/down/nothing).
		ArrayList<Double> stateDefaultActionValues = new ArrayList<Double>(Collections.nCopies(state.getActionSize(), 0.0));
		ArrayList<Double> newStateDefaultActionValues = new ArrayList<Double>(Collections.nCopies(newState.getActionSize(), 0.0));
		
		// Make entries in qTable if they don't exist yet
		this.qTable.putIfAbsent(state.hash(), stateDefaultActionValues);
		this.qTable.putIfAbsent(newState.hash(), newStateDefaultActionValues);
		
		// Perform update
		Double Qnext = this.maxActionValue(newState);
		Double newActionValue = this.qFunction(state, action) + this.alpha * (reward + this.gamma * Qnext - this.qFunction(state, action));
		this.qTable.get(state.hash()).set(action, newActionValue);
		
		
		
		// Initialize eligibility trace if it is non-existing or empty
		
		// If 
		
		// Add current state, action, reward to eligibility trace
		
		// If terminal state update all states in eligibility trace
	}
	public void setHyperparameters(StrategyParameters properties) {
		super.setHyperparameters(properties);
		this.lambda = properties.getValueAsDouble("lambda");
	}
}
