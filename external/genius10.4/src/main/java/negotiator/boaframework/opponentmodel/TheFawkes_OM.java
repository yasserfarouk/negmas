package negotiator.boaframework.opponentmodel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.timeline.Timeline;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import genius.core.utility.EvaluatorDiscrete;
import negotiator.boaframework.offeringstrategy.anac2013.TheFawkes.JWave_Daubechie;
import negotiator.boaframework.offeringstrategy.anac2013.TheFawkes.JWave_DiscreteWaveletTransform;
import negotiator.boaframework.offeringstrategy.anac2013.TheFawkes.SmoothingCubicSpline;

/**
 * Opponent Model
 */
public final class TheFawkes_OM extends OpponentModel {
	public double timeIndexFactor;
	public double[] chi, ratio;
	public SmoothingCubicSpline alpha;
	// The weight that is added each turn to the issue weights which changed; a
	// trade-off between concession speed and accuracy.
	private double learnCoef;
	// The value that is added to a value if it is found; determines how fast
	// the value weights converge.
	private int learnValueAddition;
	private int amountOfIssues;
	private double maxOpponentBidTimeDiff;

	@Override
	public void init(NegotiationSession nSession, Map<String, Double> params) {
		super.init(nSession, params);

		if (nSession.getTimeline().getType() == Timeline.Type.Time) { // Estimate
																		// around
																		// 1 bid
																		// per
																		// 0.01
																		// seconds
																		// (and
																		// otherwise
																		// abstract
																		// to
																		// this)
			this.timeIndexFactor = 100 * nSession.getTimeline().getTotalTime();
		} else { // Rounds; at most 1 bid per round is possbile, so this is
					// great :)
			this.timeIndexFactor = nSession.getTimeline().getTotalTime();
		}

		this.learnCoef = 0.8;
		this.learnValueAddition = 1;

		File frequencyParams = new File("g3_frequencyparams.txt");
		if (frequencyParams.exists() && frequencyParams.canRead()) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(frequencyParams));
				this.learnCoef = Double.parseDouble(reader.readLine());
				this.learnValueAddition = Integer.parseInt(reader.readLine());
				reader.close();
			} catch (IOException io) {
				io.printStackTrace();
			}
		}

		initializeFrequencyModel();
	}

	private void initializeFrequencyModel() {
		this.opponentUtilitySpace = (AdditiveUtilitySpace) this.negotiationSession.getUtilitySpace().copy();
		this.amountOfIssues = this.opponentUtilitySpace.getDomain().getIssues().size();
		double commonWeight = 1 / (double) this.amountOfIssues;
		for (Map.Entry<Objective, Evaluator> ev : this.opponentUtilitySpace.getEvaluators()) { // set
																								// the
																								// issue
																								// weights
			this.opponentUtilitySpace.unlock(ev.getKey());
			ev.getValue().setWeight(commonWeight);
			for (ValueDiscrete vd : ((IssueDiscrete) ev.getKey()).getValues()) { // set
																					// all
																					// value
																					// weights
																					// to
																					// one
																					// (they
																					// are
																					// normalized
																					// when
																					// calculating
																					// the
																					// utility)
				EvaluatorDiscrete ed = (EvaluatorDiscrete) ev.getValue();
				try {
					ed.setEvaluation(vd, 1);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * Determines the difference between bids. For each issue, it is determined
	 * if the value changed. If this is the case, a 1 is stored in a hashmap for
	 * that issue, else a 0.
	 *
	 * @param a
	 *            bid of the opponent
	 * @param another
	 *            bid
	 * @return
	 */
	private HashMap<Integer, Integer> determineDifference(Bid first, Bid second) {
		HashMap<Integer, Integer> diff = new HashMap<Integer, Integer>();
		for (Issue i : this.opponentUtilitySpace.getDomain().getIssues()) {
			try {
				ValueDiscrete firstValue = (ValueDiscrete) first.getValue(i.getNumber());
				ValueDiscrete secondValue = (ValueDiscrete) second.getValue(i.getNumber());
				diff.put(i.getNumber(), firstValue.equals(secondValue) ? 0 : 1);
			} catch (Exception e) {
				diff.put(i.getNumber(), 0);
				e.printStackTrace();
			}
		}
		return diff;
	}

	@Override
	public double getBidEvaluation(Bid bid) {
		try {
			return this.opponentUtilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}
	}

	public double getMaxOpponentBidTimeDiff() {
		return Math.min(0.1, this.maxOpponentBidTimeDiff);
	}

	@Override
	public void updateModel(Bid bid, double time) {
		BidHistory opponentHistory = this.negotiationSession.getOpponentBidHistory().sortToTime(); // new->old
		if (opponentHistory.size() < 2) {
			return;
		}
		Bid previousBid = opponentHistory.getHistory().get(1).getBid();
		HashMap<Integer, Integer> lastDiffSet = determineDifference(bid, previousBid);
		int numberOfUnchanged = 0;
		for (Integer i : lastDiffSet.keySet()) { // count the number of changes
													// in value
			if (lastDiffSet.get(i) == 0) {
				numberOfUnchanged++;
			}
		}

		// This is the value to be added to weights of unchanged issues before
		// normalization.
		// Also the value that is taken as the minimum possible weight,
		// (therefore defining the maximum possible also).
		double goldenValue = this.learnCoef / (double) this.amountOfIssues;
		// The total sum of weights before normalization.
		double totalSum = 1 + (goldenValue * numberOfUnchanged);
		// The maximum possible weight
		double maximumWeight = 1 - ((this.amountOfIssues * goldenValue) / totalSum);

		for (Integer i : lastDiffSet.keySet()) { // re-weighing issues while
													// making sure that the sum
													// remains 1
			if (lastDiffSet.get(i) == 0 && this.opponentUtilitySpace.getWeight(i) < maximumWeight) {
				this.opponentUtilitySpace.setWeight(
						this.opponentUtilitySpace.getDomain().getObjectivesRoot().getObjective(i),
						(this.opponentUtilitySpace.getWeight(i) + goldenValue) / totalSum);
			} else {
				this.opponentUtilitySpace.setWeight(
						this.opponentUtilitySpace.getDomain().getObjectivesRoot().getObjective(i),
						this.opponentUtilitySpace.getWeight(i) / totalSum);
			}
		}
		for (Map.Entry<Objective, Evaluator> ev : this.opponentUtilitySpace.getEvaluators()) { // Then
																								// for
																								// each
																								// issue
																								// value
																								// that
																								// has
																								// been
																								// offered
																								// last
																								// time,
																								// a
																								// constant
																								// value
																								// is
																								// added
																								// to
																								// its
																								// corresponding
																								// ValueDiscrete.
																								// cast
																								// issue
																								// to
																								// discrete
																								// and
																								// retrieve
																								// value.
																								// Next,
																								// add
																								// constant
																								// learnValueAddition
																								// to
																								// the
																								// current
																								// preference
																								// of
																								// the
																								// value
																								// to
																								// make
																								// it
																								// more
																								// important
			EvaluatorDiscrete ed = (EvaluatorDiscrete) ev.getValue();
			IssueDiscrete id = (IssueDiscrete) ev.getKey();
			try {
				ValueDiscrete vd = (ValueDiscrete) bid.getValue(id.getNumber());
				ed.setEvaluation(vd, this.learnValueAddition + ed.getEvaluationNotNormalized(vd));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		// Paper step 0 (formulas 1 and 2) are already in the Genius framework
		// (utility and discount)
		// Paper step 1 is to record the timestamps and utilities of opponent
		// bids. This is already done as well (session.getOpponentBidHistory)
		// Paper step 2 (formulas 3, 4, and 5) is the DiscreteWaveletTransform
		// of the OpponentBidHistory, which is done below:
		HashMap<Integer, Double> map = new HashMap<Integer, Double>(opponentHistory.size());
		int maxtime = -1;
		double prevBidTime = opponentHistory.getLastBidDetails().getTime();
		for (BidDetails biddetail : opponentHistory.getHistory()) {
			double diff = prevBidTime - biddetail.getTime();
			if (diff > this.maxOpponentBidTimeDiff) {
				this.maxOpponentBidTimeDiff = diff;
			}
			prevBidTime = biddetail.getTime();

			int bidtime = (int) Math.floor(biddetail.getTime() * this.timeIndexFactor); // map
																						// relative
																						// the
																						// time
																						// double
																						// to
																						// an
																						// integer
																						// index
			map.put(bidtime, biddetail.getMyUndiscountedUtil());
			if (bidtime > maxtime) { // keep track of the maximum
				maxtime = bidtime;
			}
		}

		this.chi = new double[maxtime + 1];
		for (int key : map.keySet()) {
			this.chi[key] = map.get(key);
			// Group3_Agent.debug( "CHI[" + key + "] = " + this.chi[key] );
		}
		JWave_DiscreteWaveletTransform dwt = new JWave_DiscreteWaveletTransform(new JWave_Daubechie()); // This
																										// Daubechie
																										// is
																										// of
																										// order
																										// 8
																										// (paper
																										// uses
																										// order
																										// 10)
		double[] decomposition = dwt.forwardWavelet(this.chi); // Do the
																// discrete
																// wavelet
																// transform on
																// the opponent
																// bid history

		// Paper step 3 (formula 6) is to run a cubic smoothing spline over the
		// decomposition (with a smoothing parameter)
		int N = decomposition.length;
		double x[] = new double[N + 1];
		double y[] = new double[N + 1];
		for (int i = 0; i < N; i++) { // Prepare the input required for the
										// cubic smoothing
			x[i] = (double) i;
			y[i] = decomposition[i];
		}

		this.alpha = new SmoothingCubicSpline(x, y, 1e-16); // Do the actual
															// cubic smoothing
															// spline
		this.ratio = new double[maxtime + 1];
		for (int i = 0; i <= maxtime; i++) { // Calculate the ratio between the
												// original and the smoothed
												// version, as required
			double currentRatio = this.alpha.evaluate(i) / this.chi[i];
			if (Double.isInfinite(currentRatio)) { // Fix edge-cases (-inf,+inf,
													// and NaN)
				currentRatio = 0;
			} else if (Double.isNaN(currentRatio)) {
				currentRatio = 1;
			}
			this.ratio[i] = currentRatio;
			// Group3_Agent.debug( "RATIO(" + i + ") = " + this.ratio[i] );
		}
	}
}
