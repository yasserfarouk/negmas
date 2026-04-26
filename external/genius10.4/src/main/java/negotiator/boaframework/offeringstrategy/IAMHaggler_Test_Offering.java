package negotiator.boaframework.offeringstrategy;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import agents.Jama.Matrix;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.timeline.DiscreteTimeline;
import genius.core.tournament.TournamentConfiguration;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.offeringstrategy.anac2011.iamhaggler2011.RandomBidCreator;
import negotiator.boaframework.opponentmodel.IAMHagglerOpponentConcessionModel;

public class IAMHaggler_Test_Offering extends OfferingStrategy {

	private IAMHagglerOpponentConcessionModel concessionModel;
	protected RandomBidCreator bidCreator;
	private int amountOfSamples;
	private BidDetails MAX_UTILITY_BID;
	private Matrix variances;
	private Matrix means;

	public IAMHaggler_Test_Offering() {
	}

	public IAMHaggler_Test_Offering(NegotiationSession negoSession, OpponentModel model, OMStrategy oms)
			throws Exception {
		init(negoSession, model, oms, null);
	}

	@Override
	public void init(NegotiationSession negotiationSession, OpponentModel opponentModel, OMStrategy omStrategy,
			Map<String, Double> parameters) throws Exception {
		super.init(negotiationSession, opponentModel, omStrategy, parameters);
		this.negotiationSession = negotiationSession;
		double amountOfRegressions;
		if (parameters.containsKey("r")) {
			amountOfRegressions = parameters.get("r");
		} else {
			amountOfRegressions = 10;
			System.out.println("Using default " + amountOfRegressions + " for amount of regressions.");
		}
		if (parameters.containsKey("s")) {
			double value = parameters.get("s");
			amountOfSamples = (int) value;
		} else {
			amountOfSamples = TournamentConfiguration.getIntegerOption("deadline", 10) / 2;
			System.out.println("Using default " + amountOfSamples + " for amount of samples.");
		}

		concessionModel = new IAMHagglerOpponentConcessionModel((int) amountOfRegressions,
				(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(), amountOfSamples);
		bidCreator = new RandomBidCreator();
		MAX_UTILITY_BID = negotiationSession.getMaxBidinDomain();
		SortedOutcomeSpace outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		negotiationSession.setOutcomeSpace(outcomespace);
	}

	@Override
	public BidDetails determineOpeningBid() {
		if (!negotiationSession.getOpponentBidHistory().isEmpty()) {
			double myUndiscountedUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
					.getMyUndiscountedUtil();
			double time = negotiationSession.getTime();
			concessionModel.updateModel(myUndiscountedUtil, time);
			System.out.println(
					"IAMHagglerOpponentConcessionModel initialized with u = " + myUndiscountedUtil + ", t = " + time);

		}
		return MAX_UTILITY_BID;
	}

	@Override
	public BidDetails determineNextBid() {

		double myUndiscountedUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails()
				.getMyUndiscountedUtil();
		double time = negotiationSession.getTime();
		int round = ((DiscreteTimeline) negotiationSession.getTimeline()).getRound();
		concessionModel.updateModel(myUndiscountedUtil, time);

		variances = concessionModel.getVariance();
		means = concessionModel.getMeans();
		if (negotiationSession.getTime() > 0.5) {

			StringWriter variancesWriter = new StringWriter();
			PrintWriter variancesPrintWriter = new PrintWriter(variancesWriter);
			variances.print(variancesPrintWriter, 10, 4);
			System.out.println("variances: " + variancesWriter.getBuffer().toString());

			StringWriter meanWriter = new StringWriter();
			PrintWriter meanPrintWriter = new PrintWriter(meanWriter);
			means.print(meanPrintWriter, 10, 4);
			System.out.println("means: " + meanWriter.getBuffer().toString());
		}

		// System.out.println("opponentBidUtil: " + opponentUtility);

		// System.out.println("Prediction: " + prediction.toString());

		// System.out.println("Prediction Mean Col: " +
		// means.getColumnDimension());
		// System.out.println("Prediction Mean Row: " +
		// means.getRowDimension());

		// System.out.println("Prediction variances Col: " +
		// variances.getColumnDimension());
		// System.out.println("Prediction variances Row: " +
		// variances.getRowDimension());

		// System.out.println();
		// System.out.println("Round " + round + (variances == null ?
		// ". Estimates still null" : ""));
		// System.out.println("model has been updated with u = " +
		// myUndiscountedUtil + ", at t = " + time +
		// " (which was offered in round " + (round - 1) + ").");
		// if(variances != null){
		//
		// DecimalFormat formatter = new DecimalFormat("#.########");
		// DecimalFormatSymbols dfs = new DecimalFormatSymbols();
		// dfs.setDecimalSeparator('.');
		// formatter.setDecimalFormatSymbols(dfs);
		//
		//
		// System.out.println("Current time\tCurrent utility\tPrediction for
		// time\tMean\tVariance\t2 SD\tMean\tMean - 2SD\tMean + 2SD");
		//
		// for (int i = 0; i <= amountOfSamples; i++)
		// {
		// double var = variances.get(i, 0);
		// double sd = Math.sqrt(var);
		// double mean = means.get(i, 0);
		// double predForTime = ((double) i / (double) amountOfSamples);
		//
		// System.out.println(time + "\t" + myUndiscountedUtil + "\t" +
		// predForTime + "\t" + mean + "\t" + formatter.format(var) + "\t" + (2
		// * sd) + "\t"
		// + mean + "\t" + (mean - 2*sd) + "\t" + (mean + 2*sd));
		// }
		// }

		return MAX_UTILITY_BID;
	}

	public Matrix getMeans() {
		return means;
	}

	public Matrix getVariances() {
		return variances;
	}

	public int getAmountOfSamples() {
		return amountOfSamples;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("r", 36.0, "Amount of regressions"));
		set.add(new BOAparameter("s", 100.0, "Amount of time samples"));

		return set;
	}

	@Override
	public String getName() {
		return "Other - IAMHaggler Test Offering";
	}
}
