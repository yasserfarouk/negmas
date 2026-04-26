package agents.anac.y2014.Aster;

import java.util.ArrayList;

import genius.core.Bid;
import genius.core.utility.AbstractUtilitySpace;

public class GetAcceptProb {
	private AbstractUtilitySpace utilitySpace;
	private double target;
	private double bidTarget;
	private double sum;
	private double sum2;
	private double prob;
	private double deviation;
	private double estimateMax;
	private int rounds;
	private double tremor;
	private double minimumConcessionUtility;
	private double minimum;
	private double maximumConcessionUtility;
	private double maximum;
	private double alpha;
	private double diff;
	private double alphaInit;
	private double estrangement;
	private boolean firstFlag;
	private ArrayList<Double> oppUtilHistory;
	private ArrayList<Double> oppShort_EMA;
	private ArrayList<Double> oppLong_EMA;
	private static final int EMA_SHORT = 26;
	private static final int EMA_LONG = 52;

	public GetAcceptProb(AbstractUtilitySpace utilitySpace) {
		this.utilitySpace = utilitySpace;
		this.target = 1.0D;
		this.bidTarget = 1.0D;
		this.sum = 0.0D;
		this.sum2 = 0.0D;
		this.prob = 0.0D;
		this.rounds = 0;
		this.tremor = 3.0D;
		this.minimumConcessionUtility = 0.1D;
		this.maximumConcessionUtility = 1.0D;
		this.maximum = 1.0D;
		this.deviation = 0.0D;
		this.estimateMax = 0.0D;
		this.alphaInit = 1.0D;
		this.estrangement = 1.0D;
		this.firstFlag = false;
		this.oppUtilHistory = new ArrayList<Double>();
		this.oppShort_EMA = new ArrayList<Double>();
		this.oppLong_EMA = new ArrayList<Double>();
		this.checkDiscountFactor();
	}

	/**
	 * Calculate Accept Probability
	 *
	 */
	public double getAcceptProbability(Bid offeredBid, double oppMaxUtil,
			double time) throws Exception {
		double unDiscountedOfferedUtil = utilitySpace.getUtility(offeredBid);
		double offeredUtility = utilitySpace.getUtilityWithDiscount(offeredBid,
				time);
		double weight, mean, variance, rand, alpha, beta, preTarget, preTarget2, ratio, ratio2;
		double utilityEval, satisfy;

		checkException(offeredUtility, time); // checkException

		if (offeredUtility > 1.0D) {
			offeredUtility = 1.0D;
		}

		// Historyã�«è¿½åŠ 
		oppUtilHistory.add(unDiscountedOfferedUtil);

		if (oppUtilHistory.size() > EMA_SHORT) {
			double emaUtility;
			if (oppShort_EMA.isEmpty()) {
				oppShort_EMA.add(calcN_EMA_init(oppUtilHistory, EMA_SHORT));
			} else {
				emaUtility = oppShort_EMA.get(oppShort_EMA.size() - 1);
				oppShort_EMA.add(calcN_EMA(emaUtility, unDiscountedOfferedUtil,
						EMA_SHORT));
			}
			if (oppUtilHistory.size() > EMA_LONG) {
				if (oppLong_EMA.isEmpty()) {
					oppLong_EMA.add(calcN_EMA_init(oppUtilHistory, EMA_LONG));
				} else {
					emaUtility = oppLong_EMA.get(oppLong_EMA.size() - 1);
					oppLong_EMA.add(calcN_EMA(emaUtility,
							unDiscountedOfferedUtil, EMA_LONG));
					estrangement = calcEMAEstrangement();
				}
			}
		}

		// é‡�ã�¿ä»˜ã�‘
		if ((this.estimateMax != 0.0) && (offeredUtility > this.estimateMax)) {
			weight = offeredUtility / estimateMax;
			offeredUtility *= weight;
		} else {
			weight = 1.0;
		}

		// å�ˆè¨ˆå€¤
		sum += offeredUtility;
		// åŠ¹ç”¨å€¤ã�®äºŒä¹—ã�®å�ˆè¨ˆ
		sum2 += offeredUtility * offeredUtility;
		// ãƒ©ã‚¦ãƒ³ãƒ‰æ•°
		rounds += weight;

		// ï¼ˆåŠ é‡�ï¼‰å¹³å�‡å€¤
		mean = sum / rounds;

		// åˆ†æ•£
		variance = sum2 / rounds - mean * mean;
		// ç›¸æ‰‹ã�®æŽ¨å®šè¡Œå‹•å¹…
		deviation = Math.sqrt(variance * 12.0D);

		if (Double.isNaN(deviation)) {
			deviation = 0.0;
		}

		// ç�¾åœ¨æŽ¨å®šã�•ã‚Œã‚‹ã€�ç›¸æ‰‹ã�‹ã‚‰å¼•ã��å‡ºã�›ã‚‹æœ€å¤§åŠ¹ç”¨å€¤
		estimateMax = mean + (1.0D - mean) * deviation;
		double diffest = oppMaxUtil - estimateMax;
		if (diffest > 0) {
			estimateMax *= 1.0D + diffest;
		}

		if (firstFlag) {
			estimateMax = .990 * estrangement;
		}

		// æŽ¥è¿‘é–¢æ•°
		alpha = alphaInit + tremor + (10.0D * mean) - (2.0D * tremor * mean);
		rand = minimum + (maximum - minimum) * Math.random();
		beta = alpha + rand * tremor - tremor / 2.0D;

		// æŽ¥è¿‘é–¢æ•°ã‚’åŸºã�«è¨ˆç®—
		preTarget = calcPreTarget(alpha, time);
		preTarget2 = calcPreTarget(beta, time);

		// ç›¸æ‰‹ã�¨ã�®è­²æ­©æ¯”çŽ‡
		ratio = calcRatio(preTarget);
		ratio2 = calcRatio(preTarget2);

		// è£œæ­£
		target = calcTarget(ratio, preTarget, time, oppMaxUtil, false);
		bidTarget = calcTarget(ratio2, preTarget2, time, oppMaxUtil, true)
				* estrangement;
		if (bidTarget > 1.0D) {
			bidTarget = 0.99D;
		}

		utilityEval = offeredUtility - oppMaxUtil;
		satisfy = offeredUtility - target;
		prob = Math.pow(time, alpha) / alpha + utilityEval + satisfy;

		if ((prob < 0.1D) || (Double.isNaN(prob))) {
			prob = 0.0D;
		}

		return prob;
	}

	/**
	 * getCurrentBidTarget
	 *
	 */
	public double getCurrentBidTarget() {
		return this.bidTarget;
	}

	public double getEstimateMax() {
		return this.estimateMax;
	}

	/**
	 * åˆ�å›žè¨­å®š
	 * 
	 * @param sessionNr
	 * @param maxConcessionUtility
	 * @param opponentMinUtility
	 */
	public void setTargetParam(int sessionNr, double maxConcessionUtility,
			double opponentMinUtility) {
		// åˆ�å›žã�¯å¼·æ°—
		if (sessionNr == 0) {
			this.firstFlag = true;
			this.alpha = 15;
			this.diff = 1.0;
		} else {
			this.alpha = 15;
			this.diff = 0.5 * (0.5 + utilitySpace.getDiscountFactor());
			this.maximumConcessionUtility = 1.0D;
			this.minimumConcessionUtility = opponentMinUtility;
		}
	}

	public void updateMinConcessionUtil(double minConcessionUtil) {
		this.minimumConcessionUtility = minConcessionUtil;
	}

	private double calcPreTarget(double x, double t) {
		return 1.0D - Math.pow(t, x) * (1.0D - this.estimateMax);
	}

	private double calcRatio(double preTarget) {
		double preRatio = (this.deviation + this.minimumConcessionUtility)
				/ (1.0D - preTarget);

		if ((Double.isNaN(preRatio)) || (preRatio > 2.0D)) {
			return 2.0D;
		} else {
			return preRatio;
		}
	}

	private double calcTarget(double ratio, double preTarget, double t,
			double oppMax, boolean checkdf) {
		double target = (ratio * preTarget + maximumConcessionUtility - ratio);
		double pt;
		if ((checkdf) && (utilitySpace.getDiscountFactor() < 1.0)) {
			pt = 1.0 / (1.0 + Math.exp(-alpha * (t - diff)));
		} else {
			pt = Math.pow(t, 3);
		}
		double m = pt * -300.0D + 400.0D;
		if (target > this.estimateMax) {
			double r = target - this.estimateMax;
			double f = 1.0D / (r * r);
			double app;

			if ((f > m) || (Double.isNaN(f))) {
				f = m;
			}
			app = r * f / m;
			target -= app;
		} else {
			target = this.estimateMax;
		}

		if (Double.isNaN(target)) {
			target = 1.0D;
		}

		target = checkReservationValue(target, t);

		return target;
	}

	// EMAã�®è¨ˆç®—ï¼ˆåˆ�å›žï¼‰
	private double calcN_EMA_init(ArrayList<Double> oppUtilHist, int n) {
		double ave = 0.0D;
		for (int i = 0; i < n; i++) {
			ave += oppUtilHist.get(i);
		}
		return ave / n;
	}

	// EMAã�®è¨ˆç®—
	private double calcN_EMA(double prevutil, double util, int n) {
		return prevutil + (2.0D / (1.0D + (double) n)) * (util - prevutil);
	}

	// EMA2ç·šã‚«ã‚¤ãƒªçŽ‡
	private double calcEMAEstrangement() {
		int size = oppLong_EMA.size() - 1;
		double emaShort = oppShort_EMA.get(size + EMA_SHORT);
		double emaLong = oppLong_EMA.get(size);
		double estrangement = 1.0 + ((emaShort - emaLong) / emaLong);
		if (estrangement > 1.0) {
			estrangement = 1.0;
		}
		return estrangement;
	}

	private double checkReservationValue(double target, double t) {
		double reservationValue = utilitySpace
				.getReservationValueWithDiscount(t);

		if (!Double.isNaN(reservationValue)) {
			if (target < reservationValue) {
				return reservationValue;
			}
		}
		return target;
	}

	private void checkException(double oUtility, double t) throws Exception {
		if ((oUtility < 0.0D) || (oUtility > 1.05D)) {
			throw new Exception("utility " + oUtility + " outside [0,1]");
		}
		if ((t < 0.0D) || (t > 1.0D)) {
			throw new Exception("time " + t + " outside [0,1]");
		}
	}

	private void checkDiscountFactor() {
		double discountfactor = utilitySpace.getDiscountFactor();

		if (discountfactor < 1.0) {
			this.minimum = 0.5;
		} else {
			this.minimum = 0.7;
		}
	}
}
