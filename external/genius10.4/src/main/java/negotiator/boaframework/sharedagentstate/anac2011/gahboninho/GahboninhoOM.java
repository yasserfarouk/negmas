package negotiator.boaframework.sharedagentstate.anac2011.gahboninho;

import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;

import java.util.Random;
import java.util.TreeMap;

public class GahboninhoOM {
	AdditiveUtilitySpace US;
	private final boolean TEST_EQUIVALENCE = false;
	private Random random500;
	private Random random600;

	/**
	 * provides prediction of how important each dispute element is, considering
	 * opponent's Behavior and "obsessions"
	 */

	// alternative ideas for calculating utility:
	// using k-nearest neigough, since opponent may use the same bid many times
	public class IssuePrediction // represents one Issue of the bid (e.g.
									// Screen-Size/Brand)
	{
		// the following tells how preferable each option is

		class NumericValues implements GahbonValueType {
			private final int DiscretizationResolution = 21;

			private double TranslateValue(Value V) {
				if (V instanceof ValueInteger)
					return ((ValueInteger) V).getValue();

				return ((ValueReal) V).getValue();
			}

			int[] ValueFrequencies;

			private int GetFrequencyIndex(double V) {
				return (int) ((V - MinValue) / (MaxValue / (DiscretizationResolution - 1)));
			}

			double MinValue;
			double MaxValue;

			boolean isFirstValue = true;
			double FirstValue; // we assume that if FirstValue == MaxValue, then
								// MaxValue has max utility

			double BidFrequencyIndexSum = 0;
			double NormalizedVariance; // 0 to 1
			int OpponentBidCountToConsider = 0;

			// this method learns from opponent's choices
			public void UpdateImportance(
					Value OpponentBid /*
										 * value of this Issue as received from
										 * opponent
										 */) {
				++OpponentBidCountToConsider;
				double BidValue = TranslateValue(OpponentBid);
				if (isFirstValue) {
					isFirstValue = false;

					// choose if the highest utility for the opponent is
					// max-value or min-value
					if ((MaxValue - BidValue) >= (BidValue - MinValue))
						FirstValue = MaxValue;
					else
						FirstValue = MinValue;
				}

				++ValueFrequencies[GetFrequencyIndex(BidValue)];
				BidFrequencyIndexSum += GetFrequencyIndex(BidValue);

				double AverageFrequencyIndex = BidFrequencyIndexSum / OpponentBidCountToConsider;

				NormalizedVariance = 0;
				for (int i = 0; i < DiscretizationResolution; ++i) {
					double Distance = (AverageFrequencyIndex - i) / (DiscretizationResolution - 1);
					NormalizedVariance += (double) (this.ValueFrequencies[i]) * Distance * Distance;
				}

				// NormalizedVariance /= BidFrequencyIndexSum;
			}

			public double GetNormalizedVariance() {
				return NormalizedVariance;
			}

			public int GetUtilitiesCount() {
				return this.DiscretizationResolution;
			}

			public double GetExpectedUtilityByValue(Value V) {
				return Math.min(1, Math.max(0, 1 - Math.abs(TranslateValue(V) - FirstValue)));
			}

			public void INIT(genius.core.issue.Issue I) {
				ValueFrequencies = new int[this.DiscretizationResolution];
				if (I.getType() == ISSUETYPE.INTEGER) {
					IssueInteger II = (IssueInteger) I;
					MaxValue = II.getUpperBound();
					MinValue = II.getLowerBound();
				} else {
					IssueReal RI = (IssueReal) I;
					MaxValue = RI.getUpperBound();
					MinValue = RI.getLowerBound();
				}

			}

		}

		class DiscreteValues implements GahbonValueType {
			TreeMap<ValueDiscrete, Integer> OptionIndexByValue = null;
			TreeMap<Integer, ValueDiscrete> ValueByOptionIndex = null;

			int MostImportantValueOccurrencesAndBonuses = 1;

			// TODO: make this funtion better!
			// OptionOccurrencesCountWithoutSundayPreference's bonus
			// should not be constant
			int TotalImportanceSum = 0; // sum of all importances

			private double GetOptionImportance(int OptionIndex) {
				int FirstOptionBonus = UpdatesCount / (OptionOccurrencesCountByIndex.length); // the
																								// first
																								// choice
																								// always
																								// gets
																								// a
																								// large
																								// bonus,
																								// so
																								// this
																								// may
																								// minimize
																								// "noises

				if (FirstIterationOptionIndex == OptionIndex) {
					return OptionOccurrencesCountByIndex[OptionIndex] + FirstOptionBonus;
				}

				return OptionOccurrencesCountByIndex[OptionIndex];

			}

			int UpdatesCount = 0;
			int FirstIterationOptionIndex = -1; // First Iteration has much more
												// weight than other iterations,
												// since it is reasonable to
												// assume it indicates
												// opponent's optimal option
			int[] OptionOccurrencesCountByIndex = null; // Tells how many times
														// each option was
														// chosen by opponent
			TreeMap<Integer, Integer> OptionIndexByImportance = new TreeMap<Integer, Integer>();
			double IssueImportanceRankVariance = 0; // normalized ( 0 to 1,
													// where 1 is maximum
													// variance)

			// this method learns from opponent's choices
			public void UpdateImportance(
					Value OpponentBid /*
										 * value of this Issue as received from
										 * opponent
										 */) {
				++UpdatesCount;

				Integer incommingOptionIndex = OptionIndexByValue.get(OpponentBid);
				if (-1 == FirstIterationOptionIndex)
					FirstIterationOptionIndex = incommingOptionIndex;
				++OptionOccurrencesCountByIndex[incommingOptionIndex];

				// let OptionIndexByOccurrencesCount sort the options by their
				// importance rank:
				OptionIndexByImportance.clear();

				MostImportantValueOccurrencesAndBonuses = 0;
				TotalImportanceSum = 0;
				for (int OptionIndex = 0; OptionIndex < OptionOccurrencesCountByIndex.length; ++OptionIndex) {
					int OptionImportance = (int) GetOptionImportance(OptionIndex);
					MostImportantValueOccurrencesAndBonuses = Math.max(MostImportantValueOccurrencesAndBonuses,
							OptionImportance);

					OptionIndexByImportance.put(OptionImportance, OptionIndex);
					TotalImportanceSum += OptionImportance;

				}

				// now calculate how easily the opponent gives up his better
				// options in this issue:
				double AverageImportanceRank = 0;
				int currentOptionRank = 0; // highest rank is 0
				for (Integer currentOptionIndex : OptionIndexByImportance.values()) {
					AverageImportanceRank += currentOptionRank * GetOptionImportance(currentOptionIndex);
					++currentOptionRank;
				}
				AverageImportanceRank /= TotalImportanceSum;

				IssueImportanceRankVariance = 0;
				currentOptionRank = 0; // highest rank is 0
				for (Integer currentOptionIndex : OptionIndexByImportance.values()) {
					double CurrentOptionDistance = (AverageImportanceRank - currentOptionRank)
							/ OptionOccurrencesCountByIndex.length; // divide by
																	// option
																	// count to
																	// normalized
																	// distances

					IssueImportanceRankVariance += OptionOccurrencesCountByIndex[currentOptionIndex] * // Occurrence
																										// count
																										// of
																										// current
																										// option
							(CurrentOptionDistance * CurrentOptionDistance); // variance
																				// of
																				// current
																				// option

					++currentOptionRank;
				}

				IssueImportanceRankVariance /= TotalImportanceSum;
			}

			public double GetNormalizedVariance() {
				return IssueImportanceRankVariance;
			}

			public int GetUtilitiesCount() {
				return ValueByOptionIndex.size();
			}

			public double GetExpectedUtilityByValue(Value V) {
				int ValueIndex = (Integer) (OptionIndexByValue.get(V));
				return GetOptionImportance(ValueIndex) / MostImportantValueOccurrencesAndBonuses;
			}

			public void INIT(genius.core.issue.Issue I) {
				IssueDiscrete DI = (IssueDiscrete) I;
				OptionOccurrencesCountByIndex = new int[DI.getNumberOfValues()];

				Comparator<ValueDiscrete> DIComparer = new Comparator<ValueDiscrete>() {
					public int compare(ValueDiscrete o1, ValueDiscrete o2) {
						return o1.getValue().compareTo(o2.getValue());
					}
				};
				OptionIndexByValue = new TreeMap<ValueDiscrete, Integer>(DIComparer);
				ValueByOptionIndex = new TreeMap<Integer, ValueDiscrete>();

				for (int ValueIndex = 0; ValueIndex < DI.getNumberOfValues(); ++ValueIndex) {
					OptionOccurrencesCountByIndex[ValueIndex] = 0;

					ValueDiscrete V = DI.getValues().get(ValueIndex);
					OptionIndexByValue.put(V, ValueIndex);
				}

			}

		}

		public double ExpectedWeight; // depends on comparison of this issue's
										// variance and other issues'
		public GahbonValueType Issue;
		public genius.core.issue.Issue IssueBase;

		public void INIT(Issue I) {
			// check what type of issue we are talking about
			if (I instanceof IssueDiscrete) {
				IssueDiscrete DI = (IssueDiscrete) I;
				String[] values = new String[DI.getValues().size()];
				int ValueIndex = 0;
				for (ValueDiscrete v : DI.getValues())
					values[ValueIndex++] = new String(v.getValue());
				IssueBase = new IssueDiscrete(DI.getName(), DI.getNumber(), values);
				Issue = new DiscreteValues();
				Issue.INIT(I);
			} else if (I instanceof IssueReal) {
				IssueReal RI = (IssueReal) I;
				IssueBase = new IssueReal(RI.getName(), RI.getNumber(), RI.getLowerBound(), RI.getUpperBound());
				Issue = new NumericValues();
				Issue.INIT(I);
			} else if (I instanceof IssueInteger) {
				IssueInteger II = (IssueInteger) I;
				IssueBase = new IssueReal(II.getName(), II.getNumber(), II.getLowerBound(), II.getUpperBound());
				Issue = new NumericValues();
				Issue.INIT(I);
			}

		}

	}

	public IssuePrediction[] IssuesByIndex = null; // holds all Issues, by index
													// corresponding to Domain's
	public TreeMap<Integer, Integer> IPIndexByIssueNumber = new TreeMap<Integer, Integer>();
	public double TotalIssueOptionsVariance;
	private TimeLineInfo timeline;

	public GahboninhoOM(AdditiveUtilitySpace utilitySpace, TimeLineInfo timeline) {
		if (TEST_EQUIVALENCE) {
			random500 = new Random(500);
			random600 = new Random(600);
		} else {
			random500 = new Random();
			random600 = new Random();
		}

		// utilitySpace is derived (and initialized) from Agent

		this.US = utilitySpace;
		this.timeline = timeline;
		IssuesByIndex = new IssuePrediction[US.getDomain().getIssues().size()];
		List<Issue> IA = US.getDomain().getIssues();

		for (int IssueIndex = 0; IssueIndex < IA.size(); ++IssueIndex) {
			IssuesByIndex[IssueIndex] = new IssuePrediction();
			IssuesByIndex[IssueIndex].INIT(IA.get(IssueIndex));
			IPIndexByIssueNumber.put(IA.get(IssueIndex).getNumber(), IssueIndex);
		}
	}

	TreeMap<String, Bid> OpponentBids = new TreeMap<String, Bid>();

	public void UpdateImportance(
			Bid OpponentBid /*
							 * Incomming Bid as received from opponent
							 */) throws Exception {
		String bidStr = OpponentBid.toString();
		if (OpponentBids.containsKey(bidStr))
			return;
		OpponentBids.put(bidStr, OpponentBid);

		final double ZeroVarianceToMinimalVarianceWeight = 3;
		// a heuristic value that tells us that an issue with no variance
		// is 3 times as important as the issue with minimal variance

		TotalIssueOptionsVariance = 0;

		double MinimalNonZeroVariance = Double.MAX_VALUE;
		int ZeroVarianceIssueCount = 0; //
		for (int IssueIndex = 0; IssueIndex < IssuesByIndex.length; ++IssueIndex) {
			int IssueID = IssuesByIndex[IssueIndex].IssueBase.getNumber();
			IssuesByIndex[IssueIndex].Issue.UpdateImportance(OpponentBid.getValue(IssueID));

			double IssueVariance = IssuesByIndex[IssueIndex].Issue.GetNormalizedVariance();

			TotalIssueOptionsVariance += IssueVariance;

			if (0 == IssueVariance)
				++ZeroVarianceIssueCount;
			else
				MinimalNonZeroVariance = Math.min(IssueVariance, MinimalNonZeroVariance);
		}

		// we decide how important each issue is, by comparing it's variance
		// to the most important issue (with minimal variance)

		TotalIssueOptionsVariance /= MinimalNonZeroVariance * (1.0 / ZeroVarianceToMinimalVarianceWeight); // we
																											// now
																											// count
																											// importance
																											// of
																											// issue
																											// with
																											// units
		// of size (1.0 / ZeroVarianceToMinimalVarianceWeight) *
		// MinimalNonZeroVariance

		// we add one unit per Zero-Variance Issue
		TotalIssueOptionsVariance += ZeroVarianceIssueCount;

		double WeightCount = 0;

		if (TotalIssueOptionsVariance != ZeroVarianceIssueCount) // check if all
																	// weights
																	// are not
																	// the same
																	// ( all
																	// variances
																	// zero)
		{
			// zero variance issue have exactly 1 VarianceUnits,
			// next minimal variance had ZeroVarianceToMinimalVarianceWeight
			// VarianceUnits
			// other issues are weighted with same relation
			double VarianceUnit = MinimalNonZeroVariance / ZeroVarianceToMinimalVarianceWeight;

			// calculate each issue weight (each weight is 0 to 1)
			for (int IssueIndex = IssuesByIndex.length - 1; IssueIndex >= 0; --IssueIndex) {
				if (0 == IssuesByIndex[IssueIndex].Issue.GetNormalizedVariance()) {
					// if the issue has 0 variance, we give it maximum weight
					// more weight than the next (non-zero variance) important
					// issue
					IssuesByIndex[IssueIndex].ExpectedWeight = 1;
				} else
					IssuesByIndex[IssueIndex].ExpectedWeight = VarianceUnit
							/ IssuesByIndex[IssueIndex].Issue.GetNormalizedVariance();

				WeightCount += IssuesByIndex[IssueIndex].ExpectedWeight;
			}
		}

		for (int IssueIndex = IssuesByIndex.length - 1; IssueIndex >= 0; --IssueIndex) {
			// if up until now we were always given the same bid, then all
			// issues has the same importance and same variance(0)
			if (TotalIssueOptionsVariance == ZeroVarianceIssueCount)
				IssuesByIndex[IssueIndex].ExpectedWeight = 1.0 / IssuesByIndex.length;
			else
				IssuesByIndex[IssueIndex].ExpectedWeight /= WeightCount; // normalize
																			// weights
		}
	}

	public double EvaluateOpponentUtility(Bid B) throws Exception {
		double UtilitySum = 0;

		try {
			for (int IssueIndex = 0; IssueIndex < IssuesByIndex.length; ++IssueIndex) {
				int IssueID = IssuesByIndex[IssueIndex].IssueBase.getNumber();
				UtilitySum += IssuesByIndex[IssueIndex].ExpectedWeight
						* IssuesByIndex[IssueIndex].Issue.GetExpectedUtilityByValue(B.getValue(IssueID));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return UtilitySum;
	}

	TreeMap<Integer, TreeMap<String, ValueDiscrete>> ValueTranslation = new TreeMap<Integer, TreeMap<String, ValueDiscrete>>();

	public Value ImproveValue(int IssueNumber, ValueDiscrete ValToImprove) throws Exception {
		TreeMap<String, ValueDiscrete> ValueTranslator = ValueTranslation.get(IssueNumber);
		ValueDiscrete resultValue = new ValueDiscrete(ValToImprove.toString());

		if (ValueTranslator != null) {
			resultValue = ValueTranslator.get(ValToImprove.toString());
			if (resultValue != null)
				return resultValue;
		} else {
			ValueTranslator = new TreeMap<String, ValueDiscrete>();
			ValueTranslation.put(IssueNumber, ValueTranslator);
		}

		IssuePrediction IS = IssuesByIndex[IPIndexByIssueNumber.get(IssueNumber)];
		Issue I = IS.IssueBase;
		Bid tmpBid = US.getDomain().getRandomBid(random500);
		tmpBid = tmpBid.putValue(IssueNumber, ValToImprove);

		double oppUtilityWithVal = IS.Issue.GetExpectedUtilityByValue(ValToImprove);
		double utilityWithVal = US.getEvaluation(IssueNumber, tmpBid);

		if (!(I instanceof IssueDiscrete))
			return ValToImprove;

		IssueDiscrete DI = (IssueDiscrete) I;

		int size = DI.getNumberOfValues();
		for (int i = 0; i < size; i++) {
			ValueDiscrete curr = DI.getValue(i);
			tmpBid = tmpBid.putValue(IssueNumber, curr);
			double myUtilityWithCurrent = US.getEvaluation(IssueNumber, tmpBid);
			double oppUtilityWithCurrent = IS.Issue.GetExpectedUtilityByValue(curr);
			// // find a value which is not worse than valTo improve but better
			// for opponent
			if (myUtilityWithCurrent >= utilityWithVal && oppUtilityWithCurrent > oppUtilityWithVal * 1.3) {
				oppUtilityWithVal = oppUtilityWithCurrent;
				resultValue = curr;
			}
		}
		ValueTranslator.put(ValToImprove.toString(), resultValue);

		return resultValue;
	}

	public Bid ImproveBid(Bid BidToImprove) throws Exception {

		Bid resultBid = US.getDomain().getRandomBid(random600);
		for (Issue issue : US.getDomain().getIssues()) {
			try {
				if (issue.getType() == ISSUETYPE.DISCRETE)
					resultBid = resultBid.putValue(issue.getNumber(),
							ImproveValue(issue.getNumber(), (ValueDiscrete) BidToImprove.getValue(issue.getNumber())));
				else
					resultBid = resultBid.putValue(issue.getNumber(), BidToImprove.getValue(issue.getNumber()));

			} catch (Exception e) {
				try {
					resultBid = resultBid.putValue(issue.getNumber(),
							(ValueDiscrete) BidToImprove.getValue(issue.getNumber()));
				} catch (Exception E) {
					return BidToImprove;
				}
			}

		}

		return resultBid;
	}

	public TreeMap<Double, Bid> FilterBids(TreeMap<Double, Bid> Bids, int DesiredResultEntries) throws Exception {
		TreeMap<Double, Bid> resultBids = new TreeMap<Double, Bid>();
		Entry<Double, Bid> bidIter = Bids.lastEntry();

		double BestKey = bidIter.getKey();
		Bid bestBid = bidIter.getValue();
		double bestOppUtil = EvaluateOpponentUtility(bestBid);
		resultBids.put(BestKey, bestBid);

		bidIter = Bids.lowerEntry(bidIter.getKey());

		while (bidIter != null && timeline.getTime() < 0.94) {

			Bid checkedBid = bidIter.getValue();
			double checkedKey = US.getUtility(checkedBid);
			double checkedOppUtil = EvaluateOpponentUtility(checkedBid);

			if (checkedOppUtil >= bestOppUtil * 0.84) {
				resultBids.put(checkedKey, checkedBid);
				if (checkedOppUtil > bestOppUtil) {
					bestBid = checkedBid;
					BestKey = checkedKey;
					bestOppUtil = checkedOppUtil;
				}
			}

			bidIter = Bids.lowerEntry(bidIter.getKey());
		}

		// if (bestBid != null)
		// resultBids.put(BestKey, bestBid);

		if (resultBids.size() < DesiredResultEntries / 10 || resultBids.size() < 20)
			return Bids;
		return resultBids;
	}
}