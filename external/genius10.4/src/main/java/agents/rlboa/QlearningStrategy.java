package agents.rlboa;

import genius.core.StrategyParameters;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.OutcomeSpace;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.misc.Range;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class QlearningStrategy extends OfferingStrategy {

	protected HashMap<Integer, ArrayList<Double>> qTable;
	protected ArrayList<Integer> actions = new ArrayList<Integer>();
	protected int bins;
	protected Double eps;
	protected Double alpha;
	protected Double gamma;
	protected AbstractState state;
	protected String mode;
	protected int timeBins;
	protected Range minMaxBin;

	public QlearningStrategy(NegotiationSession negotiationSession, OpponentModel opponentModel) {
		super.init(negotiationSession, null);
		this.opponentModel = opponentModel;
		this.endNegotiation = false;
		this.state = State.INITIAL;
				
		OutcomeSpace outcomeSpace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		this.negotiationSession.setOutcomeSpace(outcomeSpace);
	}

	public ArrayList<Integer> getActions() {
		return this.actions;
	}

	/**
	 * @return int representing the last action taken by the strategy
	 * @throws IndexOutOfBoundsException if called before any action has been performed
	 */
	public int getLastAction() throws IndexOutOfBoundsException {
		return this.actions.get(this.actions.size() - 1);
	}
	
	public void setMinMaxBin(Range minMaxBin) {
		this.minMaxBin = minMaxBin;
	}
	
	public Range getMinMaxBin() {
		return this.minMaxBin;
	}

	protected void initQTable() {
		this.qTable = new HashMap<Integer, ArrayList<Double>>();

		// Initial state has different action space
		this.qTable.putIfAbsent(this.state.hash(), new ArrayList<Double>(Collections.nCopies(this.state.getActionSize(), 0.0)));
	}

	public void initQtable(HashMap<Integer, ArrayList<Double>> qTable) {
		if (qTable != null) {
			this.qTable = qTable;
		}
		else {
			this.initQTable();
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		// Open the negotiation with a free bid (one of N bins)
		int targetBin = this.determineOpeningBin();
		return this.pickBidInBin(targetBin);
	}

	@Override
	public BidDetails determineNextBid() {
		// HACK(?) this QlearningStrategy works for all states that represent the world in bins,
		// so we needed a way to recognize these. Therefore the interface BinnedRepresentation
		int targetBin = this.determineTargetBin(((BinnedRepresentation) this.state).getMyBin());
		return this.pickBidInBin(targetBin);
	}

	@Override
	public String getName() {
		return "Q-Offering";
	}

	/**
	 * Check if the bid falls inside the lower and upper bounds
	 * @param lower lower bound of utility (inclusive)
	 * @param upper upper bound of utility (exclusive)
	 * @param bidDetails bid to check (has util and time)
	 * @return boolean
	 */
	private boolean isInBin(double lower, double upper, BidDetails bidDetails) {
		double myUtil = bidDetails.getMyUndiscountedUtil();
		return myUtil < upper && myUtil >= lower;
	}

	/**
	 * Make the opponent model select a bid that is in the provided target bin
	 * @param targetBin index of the bin in which to pick a bid
	 * @return BidDetails of the selected bid
	 */
	protected BidDetails pickBidInBin(int targetBin) {

		double lowerBound = targetBin * this.getBinSize();
		double upperBound = lowerBound + this.getBinSize();

		// getBidsInRange behaves weirdly and returns bids that are outise of there range (false positives)
		List<BidDetails> bidsInRange = this.negotiationSession.getOutcomeSpace().getBidsinRange(new Range(lowerBound, upperBound));
		bidsInRange.removeIf( bid -> !this.isInBin(lowerBound, upperBound, bid) );

		// If there are no bids possible within this bin, recursively choose another bin by the following logic: 
		// if you conceded this round, concede further, etc.
		if (bidsInRange.isEmpty()) {

			Random random = new Random();
			int newBin = 0;
			int direction = -1;
			
			// Check if this is the opening action or not; if it is we just pick randomly
			if (this.actions.size() > 1) {
				direction = this.actions.get(this.actions.size() - 1);
			} else {
				newBin = random.nextInt(this.bins);
			}

			// conceded last time
			if (direction == 0) {
				newBin = determineTargetBin(targetBin - 1);
			}
			
			// retracted last time
			if (direction == 1) {
				newBin = determineTargetBin(targetBin + 1);
			}
			
			// stayed last time
			if (direction == 2) {
				int randomUpOrDown = random.nextBoolean() ? 1 : -1;
				newBin = determineTargetBin(targetBin + randomUpOrDown);
			}
			
			return this.pickBidInBin(newBin);
		}
		
		
		return this.maxBidForOpponent(bidsInRange);
	}

	/**
	 * This is the general action function for the RL-agent. We determine a bin by either
	 * moving up (retracting offer), doing nothing or moving down (conceding offer).
	 * @param currentBin
	 * @return
	 */
	protected int determineTargetBin(int currentBin) {
		int targetBin = currentBin;
		ArrayList<Double> defaultActionValues = new ArrayList<Double>(Collections.nCopies(this.state.getActionSize(), 0.0));

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

		// Can't go outside of the range of relevant bins.
		targetBin = Math.min(targetBin, (int) this.minMaxBin.getUpperbound());
		targetBin = Math.max(targetBin, (int) this.minMaxBin.getLowerbound());

		return targetBin;
	}

	protected int determineOpeningBin() {
		ArrayList<Double> defaultInitialActionValues = new ArrayList<Double>(Collections.nCopies(this.state.getActionSize(), 0.0));
		List<Double> qValues = this.qTable.getOrDefault(this.state.hash(), defaultInitialActionValues);
		int action = this.epsilonGreedy(qValues);
		this.actions.add(action);

		return action;
	}

	/**
	 * @param list List of doubles
	 * @return The index of the highest value in the list
	 */
	protected int indifferentArgMax(List<Double> list) {
		double maximum = Collections.max(list);

		List<Integer> maximaIdxs = new ArrayList<Integer>();

		// collect indices of all occurrences of maximum
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) == maximum) {
				maximaIdxs.add(i);
			}
		}

		// pick a random index from the list (this is the indifferent part)
		Random rnd = new Random();
		int choice = rnd.nextInt(maximaIdxs.size());

		return maximaIdxs.get(choice);
	}

	protected int epsilonGreedy(List<Double> qValues) {
		int action;

		// With probability epsilon, pick a random action (epsilon greedy)
		if (Math.random() < this.eps && this.isTraining()) {
			Random random = new Random();
			action = random.nextInt(qValues.size());
		}
		else {
			action = this.indifferentArgMax(qValues);
		}

		return action;
	}

	/**
	 * @return The number of bins in which the each utility axis is divided
	 */
	int getNBins() {
		return this.bins;
	}

	/**
	 * @return The width of the bins in which the each utility axis is divided
	 */
	protected double getBinSize() {
		return 1.0 / this.getNBins();
	}

	/**
	 * Setter for the state property
	 * @param state new {@link State}
	 *
	 */
	protected void setState(State state) {
		this.state = state;
	}

	/**
	 * Getter for the state property
	 * @return
	 */
	protected AbstractState getState() {
		return this.state;
	}

	/**
	 * Determine the bid with the highest expected utility for the opponent from a list of bids
	 * @param bids
	 * @return BidDetails with representing the maximum bid
	 */
	protected BidDetails maxBidForOpponent(List<BidDetails> bids) {
		BidDetails maxBid = null;

		for (BidDetails bid : bids) {
			if (maxBid == null) {
				maxBid = bid;
			}
			else if (this.opponentModel.getBidEvaluation(bid.getBid()) > this.opponentModel.getBidEvaluation(maxBid.getBid())) {
				maxBid = bid;
			}
		}

		return maxBid;
	}

	/**
	 * Gets called by Negotiator when a relevant negotiation event occurs
	 * @param reward
	 * @param newState
	 */
	public void observeEnvironment(double reward, AbstractState newState) {

		// Only start updating after an action is performed
		// Only update if training is enabled
		if (this.actions.size() > 0 && this.isTraining()) {
			this.updateQFuction(this.state, this.getLastAction(), reward, newState);
		}
		this.state = newState;
	}

	public HashMap<Integer, ArrayList<Double>> getQTable() {
		return this.qTable;
	}

	protected void updateQFuction(AbstractState state, int action, double reward, AbstractState newState) {
		// initialize state if it is new

		// If agent hasn't done a opening bid, initialize action values to number of bins, otherwise
		// just 3 values (up/down/nothing).
		ArrayList<Double> stateDefaultActionValues = new ArrayList<Double>(Collections.nCopies(state.getActionSize(), 0.0));
		ArrayList<Double> newStateDefaultActionValues = new ArrayList<Double>(Collections.nCopies(newState.getActionSize(), 0.0));

		// Make entries in qTable if they don't exist yet
		this.qTable.putIfAbsent(state.hash(), stateDefaultActionValues);
		this.qTable.putIfAbsent(newState.hash(), newStateDefaultActionValues);


		// To remind ourselves that the below function is correct =>
		// the gamma comes from the domain/preference profile through reward  which is discounted.
		Double gamma = 1.0;
		// Perform update
		Double Qnext = this.maxActionValue(newState);
		Double newActionValue = this.qFunction(state, action) + this.alpha * (reward + gamma * Qnext - this.qFunction(state, action));
		this.qTable.get(state.hash()).set(action, newActionValue);
	}

	/**
	 * Determine max_a Q(s,a)
	 * @param state The hash of the state for which to retrieve the max action value
	 * @return Value of optimal action given that can be taken in the provided state (0 if state is unknown)
	 */
	protected Double maxActionValue(AbstractState state) {
		return Collections.max(this.qTable.get(state.hash()));
	}

	/**
	 * Get the Q value associated with the provided (state, action) pair.
	 * @param state
	 * @param action
	 * @return
	 */
	protected Double qFunction(AbstractState state, int action) {
		ArrayList<Double> actionValues = this.qTable.get(state.hash());
		return actionValues.get(action);
	}

	public void setHyperparameters(StrategyParameters properties) {
		this.eps = properties.getValueAsDouble("epsilon");
		this.alpha = properties.getValueAsDouble("alpha");
		this.bins = (int) properties.getValueAsDouble("bins");
		this.mode = properties.getValueAsString("_mode");
		this.timeBins = (int) properties.getValueAsDouble("time_bins");
	}
	
	protected boolean isTraining() {
		return this.mode.equals("train");
	}
}
