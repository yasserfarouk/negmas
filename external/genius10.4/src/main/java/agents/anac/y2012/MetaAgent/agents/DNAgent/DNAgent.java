package agents.anac.y2012.MetaAgent.agents.DNAgent;

import java.io.IOException;
import java.util.ArrayList;
//import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;

public class DNAgent extends Agent {

	private Action actionOfPartner = null;
	private static double MINIMUM_BID_UTILITY = 0.5;

	private static int POPULATION_SIZE_DEFAULT = 300;
	private static int TOURNAMENT_SELECTION_DEFAULT = 15;
	private static int SELECTION_DEFAULT = 100;
	private static int MUTATION_DEFAULT = 100;
	private static int CROSSOVER_DEFAULT = 100;
	// private static double ALPHA_DEFAULT=0.5;

	private ArrayList<Bid> population;
	private ArrayList<Bid> partnerBids;

	private Bid referenceBid; // To compute similarity

	private int populationSize;
	private int tournamentSize;
	private int selectionSize;
	private int mutationSize;
	private int crossoverSize;

	private double currentAspirationLevel;
	private double minimumOfferedUtil;

	// private double alpha;

	// TODO: Remove all logging before submitting
	public static Logger logger;
	static {
		try {
			boolean append = true;
			FileHandler fh = new FileHandler("TestLog.log", append);
			// fh.setFormatter(new XMLFormatter());
			fh.setFormatter(new SimpleFormatter());
			logger = Logger.getLogger("TestLog");
			logger.addHandler(fh);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void init() {
		// System.err.println("debug : ----- Initialize -----");

		if (utilitySpace.getReservationValue() != null)
			MINIMUM_BID_UTILITY = utilitySpace.getReservationValue();
		populationSize = POPULATION_SIZE_DEFAULT;
		tournamentSize = TOURNAMENT_SELECTION_DEFAULT;
		selectionSize = SELECTION_DEFAULT;
		crossoverSize = CROSSOVER_DEFAULT;
		mutationSize = MUTATION_DEFAULT;
		// alpha=ALPHA_DEFAULT;
		currentAspirationLevel = MINIMUM_BID_UTILITY;
		// System.out.println("Current aspiration level: "+currentAspirationLevel);
		partnerBids = new ArrayList<Bid>();
		minimumOfferedUtil = 1.0;

		try {
			population = createInitialPopulation();
		} catch (Exception e) {
			logger.severe(e.getMessage());
			// System.out.println("Exception creating initial population:"+e.getMessage());
		}
	}

	@Override
	public String getVersion() {
		// Versions use genetic encoding. Just for fun.
		return "CCU";
	}

	public void ReceiveMessage(Action opponentAction) {
		// System.err.println("debug : ----- ReceiveMessage -----");
		actionOfPartner = opponentAction;
	}

	public Action chooseAction() {
		Action action = null;
		try {
			if (actionOfPartner == null)
				action = proposeBid();
			if (actionOfPartner instanceof Offer) {
				Bid partnerBid = ((Offer) actionOfPartner).getBid();

				partnerBids.add(partnerBid);
				double offeredUtilFromOpponent = getUtility(partnerBid);
				if (referenceBid == null)
					referenceBid = partnerBid;

				if (getUtility(referenceBid) < offeredUtilFromOpponent)
					referenceBid = partnerBid;

				if (currentAspirationLevel < offeredUtilFromOpponent) {
					currentAspirationLevel = offeredUtilFromOpponent;
					// System.out.println("Current aspiration level: "+currentAspirationLevel);
				}

				population = updatePopulation(partnerBids);

				action = proposeBid();
				Bid myBid = ((Offer) action).getBid();
				double myOfferedUtil = getUtility(myBid);
				if (myOfferedUtil < minimumOfferedUtil) {
					myOfferedUtil = minimumOfferedUtil;
					// System.out.println("Minimum offered utility: "+myOfferedUtil);
				}
				if (minimumOfferedUtil <= offeredUtilFromOpponent) {
					// System.out.println("YO: "+myOfferedUtil+
					// "EL: "+offeredUtilFromOpponent+
					// " - Gladly accept opponent's bid");
					action = new Accept(getAgentID(), partnerBid);
				}
			}
			// sleep(0.005); // just for fun
		} catch (Exception e) {
			// System.out.println("Exception in ChooseAction:"+e.getMessage());
			logger.severe(e.getMessage());
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		return action;
	}

	/**
	 * Create the initial population for the genetic algorithm
	 * 
	 * @return new ArrayList<Bid>
	 * @throws Exception
	 */
	private ArrayList<Bid> createInitialPopulation() throws Exception {
		ArrayList<Bid> newPopulation = new ArrayList<Bid>();
		for (int i = 0; i < populationSize; i++) {
			newPopulation.add(getRandomBid());
		}
		// System.out.println("Population size is "+newPopulation.size());
		return newPopulation;
	}

	private ArrayList<Bid> updatePopulation(ArrayList<Bid> partnerBids)
			throws Exception {
		ArrayList<Bid> parents = null;
		ArrayList<Bid> crossover = null;
		ArrayList<Bid> mutation = null;
		ArrayList<Bid> newPopulation = new ArrayList<Bid>();

		// Selection
		parents = performSelection(selectionSize);
		// The opponent's bid always is part of the new parents set
		// parents.add(partnerBid);

		// Crossover
		crossover = performCrossover(parents, partnerBids, crossoverSize);

		// Mutation
		mutation = performMutation(parents, mutationSize);

		// New population is the sum of the three lists
		newPopulation.addAll(parents);
		newPopulation.addAll(crossover);
		newPopulation.addAll(mutation);

		// System.out.println("Updated population. New population size: "+newPopulation.size());

		return newPopulation;
	}

	/**
	 * 
	 * @return new ArrayList<Bid>
	 * @throws Exception
	 */
	private ArrayList<Bid> performSelection(int selectionNumber)
			throws Exception {
		ArrayList<Bid> selection = new ArrayList<Bid>();
		ArrayList<Bid> sorted = new ArrayList<Bid>();

		// Sort the elements in the population
		// Insertion method
		sorted.add(population.get(0));
		for (int i = 1; i < population.size(); i++) {
			// Insert element in the appropriate position
			Bid toInsert = population.get(i);
			boolean inserted = false;
			double toInsertFitness = getFitness(toInsert);
			for (int j = 0; j < sorted.size(); j++) {
				double positionFitness = getFitness(sorted.get(j));
				if (toInsertFitness > positionFitness) {
					sorted.add(j, toInsert);
					inserted = true;
					break;
				}
			}
			if (!inserted) {
				sorted.add(toInsert);
			}

		}

		// Population is sorted; now get the first N elements
		for (int k = 0; k < selectionNumber; k++) {
			selection.add(sorted.get(k));
		}

		return selection;
	}

	private ArrayList<Bid> performCrossover(ArrayList<Bid> fathers,
			ArrayList<Bid> mothers, int crossoverNumber) throws Exception {
		ArrayList<Bid> crossover = new ArrayList<Bid>();
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random rand = new Random();

		// Randomly add elements generated from combining issue values of two
		// parents
		while (crossover.size() < crossoverNumber) {
			// Randomly choose two parents
			int indexFather = rand.nextInt(fathers.size());
			int indexMother = rand.nextInt(mothers.size());

			Bid father = fathers.get(indexFather);
			Bid mother = mothers.get(indexMother);

			Bid son = null;

			for (Issue lIssue : issues) {
				int issueIndex = lIssue.getNumber();
				// Flip a coin
				if (rand.nextBoolean()) {
					values.put(issueIndex, father.getValue(issueIndex));
				} else {
					values.put(issueIndex, mother.getValue(issueIndex));
				}

			}
			son = new Bid(utilitySpace.getDomain(), values);
			if (getUtility(son) >= currentAspirationLevel) {
				crossover.add(son);
			}

		}

		return crossover;
	}

	private ArrayList<Bid> performMutation(ArrayList<Bid> parents,
			int mutationNumber) throws Exception {
		ArrayList<Bid> mutation = new ArrayList<Bid>();
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random rand = new Random();

		// Randomly add elements generated from combining issue values of two
		// parents
		while (mutation.size() < mutationNumber) {
			// Randomly choose a parent
			Bid parent = parents.get(rand.nextInt(parents.size()));

			Bid son = null;

			// Randomly choose an issue to mutate
			int mutateIssue = rand.nextInt(issues.size());
			int currentIssue = 0;
			for (Issue lIssue : issues) {
				if (currentIssue == mutateIssue) {
					switch (lIssue.getType()) {
					case DISCRETE:
						IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
						int optionIndex = rand.nextInt(lIssueDiscrete
								.getNumberOfValues());
						values.put(lIssue.getNumber(),
								lIssueDiscrete.getValue(optionIndex));
						break;
					case REAL:
						IssueReal lIssueReal = (IssueReal) lIssue;
						int optionInd = rand.nextInt(lIssueReal
								.getNumberOfDiscretizationSteps() - 1);
						values.put(
								lIssueReal.getNumber(),
								new ValueReal(
										lIssueReal.getLowerBound()
												+ (lIssueReal.getUpperBound() - lIssueReal
														.getLowerBound())
												* (double) (optionInd)
												/ (double) (lIssueReal
														.getNumberOfDiscretizationSteps())));
						break;
					case INTEGER:
						IssueInteger lIssueInteger = (IssueInteger) lIssue;
						int optionIndex2 = lIssueInteger.getLowerBound()
								+ rand.nextInt(lIssueInteger.getUpperBound()
										- lIssueInteger.getLowerBound());
						values.put(lIssueInteger.getNumber(), new ValueInteger(
								optionIndex2));
						break;
					default:
						throw new Exception("issue type " + lIssue.getType()
								+ " not supported by SimpleAgent2");
					}
				} else {
					int issueIndex = lIssue.getNumber();
					values.put(issueIndex, parent.getValue(issueIndex));
				}
				currentIssue = currentIssue + 1;

			}
			son = new Bid(utilitySpace.getDomain(), values);
			if (getUtility(son) >= currentAspirationLevel) {
				mutation.add(son);
			}
		}

		return mutation;
	}

	private double getFitness(Bid bid) throws Exception {
		if (referenceBid == null) {
			return utilitySpace.getUtility(bid);
		} else {
			double myTime = timeline.getTime();
			double discount = utilitySpace.getDiscountFactor();
			if ((discount <= 0) || (discount >= 1))
				discount = 1;
			double beta = Math.pow(1 - myTime,
					(1.0 / (3.5 * discount * discount)));
			// System.out.println("Discountfactor: "+discount+" Time: "+myTime+" Beta: "+beta);
			return beta * utilitySpace.getUtility(bid) + (1 - beta)
					* getSimilarity(bid, referenceBid)
					/ utilitySpace.getDomain().getIssues().size();
		}
	}

	/**
	 * Generate a bid based on the current population. Uses tournament
	 * selection.
	 * 
	 * @return new Action(Bid(..)), If a problem occurs, it returns an Accept()
	 *         action.
	 */
	private Action proposeBid() {
		Bid nextBid = null;
		Random rand = new Random();
		try {
			for (int i = 0; i < tournamentSize; i++) {
				// System.out.println("Getting candidate "+i);
				int dice = rand.nextInt(population.size());
				Bid candidateBid = population.get(dice);
				// System.out.println("Comparing utilities...");
				if (nextBid == null) {
					nextBid = candidateBid;
				} else if (getFitness(candidateBid) > getFitness(nextBid)) {
					nextBid = candidateBid;
				}
			}
		} catch (Exception e) { // System.out.println("Problem generating initial bid:"+e.getMessage()+". cancelling bidding");
			// logger.severe(e.getLocalizedMessage());
			// logger.severe(e.getStackTrace().toString());
		}
		if (nextBid == null)
			return (new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid()));
		return (new Offer(getAgentID(), nextBid));
	}

	private double getSimilarity(Bid aBid, Bid anotherBid) throws Exception {
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		double similarity = 0.0;
		int issueID;
		for (Issue lIssue : issues) {
			switch (lIssue.getType()) {
			case DISCRETE:
				IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
				issueID = lIssueDiscrete.getNumber();
				if (((ValueDiscrete) aBid.getValue(issueID)).getValue().equals(
						((ValueDiscrete) anotherBid.getValue(issueID))
								.getValue())) {
					similarity += 1.0;
				}
				break;
			case REAL:
				IssueReal lIssueReal = (IssueReal) lIssue;
				issueID = lIssueReal.getNumber();
				similarity += 1 - (Math
						.abs(((ValueReal) aBid.getValue(issueID)).getValue()
								- ((ValueReal) anotherBid.getValue(issueID))
										.getValue()) / (lIssueReal
						.getUpperBound() - lIssueReal.getLowerBound()));
				break;
			case INTEGER:
				IssueInteger lIssueInteger = (IssueInteger) lIssue;
				issueID = lIssueInteger.getNumber();
				similarity += 1 - (Math.abs(((ValueInteger) aBid
						.getValue(issueID)).getValue()
						- ((ValueInteger) anotherBid.getValue(issueID))
								.getValue()) / (lIssueInteger.getUpperBound() - lIssueInteger
						.getLowerBound()));
				break;
			default:
				throw new Exception("issue type " + lIssue.getType()
						+ " not supported by SimpleAgent2");
			}
		}
		return similarity;

	}

	/**
	 * @return a random bid with high enough utility value.
	 * @throws Exception
	 *             if we can't compute the utility (eg no evaluators have been
	 *             set) or when other evaluators than a DiscreteEvaluator are
	 *             present in the util space.
	 */
	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		// createFrom a random bid with utility>MINIMUM_BID_UTIL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (2 minutes)
		// but this is just a simple agent.
		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete
							.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal
							.getNumberOfDiscretizationSteps() - 1);
					values.put(
							lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal
											.getLowerBound())
									* (double) (optionInd)
									/ (double) (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(
							optionIndex2));
					break;
				default:
					// System.out.println("Exception creating random bid");
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by SimpleAgent2");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (getUtility(bid) < MINIMUM_BID_UTILITY);

		return bid;
	}

}
