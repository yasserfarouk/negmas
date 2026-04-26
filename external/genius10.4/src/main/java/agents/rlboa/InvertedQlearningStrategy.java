package agents.rlboa;

import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class InvertedQlearningStrategy extends QlearningStrategy {
	
	public InvertedQlearningStrategy(NegotiationSession negotiationSession, OpponentModel opponentModel) {
		super(negotiationSession, opponentModel);
	}
	
	@Override
	protected void initQTable() {
		this.qTable = new HashMap<Integer, ArrayList<Double>>();
		
		// Initial state has different action space
		this.qTable.putIfAbsent(this.state.hash(), new ArrayList<Double>(Collections.nCopies(this.state.getActionSize(), 1.0)));
	}

	@Override
	public String getName() {
		return "Inverted Q-offering";
	}
	
	/**
	 * This is the general action function for the RL-agent. We determine a bin by either
	 * moving up (retracting offer), doing nothing or moving down (conceding offer).
	 * @param currentBin
	 * @return
	 */
	@Override
	protected int determineTargetBin(int currentBin) {
		int targetBin = currentBin;
		ArrayList<Double> defaultActionValues = new ArrayList<Double>(Collections.nCopies(this.state.getActionSize(), 1.0));

		List<Double> qValues = this.qTable.getOrDefault(this.state.hash(), defaultActionValues);
		int action = this.epsilonGreedy(qValues);
		this.actions.add(action);
		
		// Apply action current bin (ie. move up, down or stay)
		switch (action) {
			case 0: targetBin--;
			break;
			case 1: targetBin++;
			break;
			case 2: break;
		}
		
		// Can't go out of bounds
		// TODO: Discuss impact on learning algorithm
		targetBin = Math.min(targetBin, this.getNBins() - 1);
		targetBin = Math.max(targetBin, 0);
		
		return targetBin;
	}
	
	@Override
	protected int determineOpeningBin() {
		ArrayList<Double> defaultInitialActionValues = new ArrayList<Double>(Collections.nCopies(this.state.getActionSize(), 1.0));
		List<Double> qValues = this.qTable.getOrDefault(this.state.hash(), defaultInitialActionValues);
		int action = this.epsilonGreedy(qValues);
		this.actions.add(action);
		
		return action;
	}
	
	@Override
	protected void updateQFuction(AbstractState state, int action, double reward, AbstractState newState) {
		// initialize state if it is new
		
		// If agent hasn't done a opening bid, initialize action values to number of bins, otherwise
		// just 3 values (up/down/nothing).
		ArrayList<Double> stateDefaultActionValues = new ArrayList<Double>(Collections.nCopies(state.getActionSize(), 1.0));
		ArrayList<Double> newStateDefaultActionValues = new ArrayList<Double>(Collections.nCopies(newState.getActionSize(), 1.0));

		// Make entries in qTable if they don't exist yet
		this.qTable.putIfAbsent(state.hash(), stateDefaultActionValues);
		this.qTable.putIfAbsent(newState.hash(), newStateDefaultActionValues);
		
		// Perform update
		Double Qnext = this.maxActionValue(newState);
		Double newActionValue = this.qFunction(state, action) + this.alpha * (reward + this.gamma * Qnext - this.qFunction(state, action));
		this.qTable.get(state.hash()).set(action, newActionValue);
	}
}

