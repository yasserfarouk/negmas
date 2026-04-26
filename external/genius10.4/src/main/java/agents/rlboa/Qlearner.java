package agents.rlboa;

import java.io.FileInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;

import genius.core.Bid;
import genius.core.StrategyParameters;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.misc.Range;
import negotiator.boaframework.acceptanceconditions.other.AC_Next;
import negotiator.boaframework.omstrategy.BestBid;
import negotiator.boaframework.opponentmodel.AgentXFrequencyModel;
import negotiator.boaframework.opponentmodel.PerfectModel;

@SuppressWarnings("serial")
public class Qlearner extends RLBOAagentBilateral {

	QlearningStrategy offeringStrategy;

	@SuppressWarnings("unchecked")
	@Override
	public void agentSetup() {

		HashMap<String, Double> params = new HashMap<String, Double>();

		// Initialize opponent model
		switch (this.getStrategyParameters().getValueAsString("opponentModel")) {
		case "PerfectModel":
			opponentModel = new PerfectModel();
			break;
		case "FrequencyModel":
			opponentModel = new AgentXFrequencyModel();
			break;
		default:
			break;
		}
		opponentModel.init(negotiationSession, params);

		// Initialize offeringStrategy (is a RL-component)
		switch (this.getStrategyParameters().getValueAsString("strategy")) {
		case "QlearningStrategy":
			offeringStrategy = new QlearningStrategy(negotiationSession, opponentModel);
			break;
		case "PriorBeliefQlearningStrategy":
			offeringStrategy = new PriorBeliefQlearningStrategy(negotiationSession, opponentModel);
			break;
		case "QLambdaStrategy":
			offeringStrategy = new QLambdaStrategy(negotiationSession, opponentModel);
			break;
		}
		offeringStrategy.setHyperparameters(this.getStrategyParameters());
		
		// Initialize q-table
		String pathToQtable = this.strategyParameters.getValueAsString("_path_to_qtable");
		String filepath = this.parsePathToQtable(pathToQtable);
		HashMap<Integer, ArrayList<Double>> qTable = (HashMap<Integer, ArrayList<Double>>) this.readObjectFromFile(filepath);
		offeringStrategy.initQtable(qTable);

		// Accept if the incoming offer is higher than what you would offer yourself
		acceptConditions = new AC_Next(negotiationSession, offeringStrategy, 1, 0);

		// Opponent model strategy always selects best bid it has available
		omStrategy = new BestBid();
		omStrategy.init(negotiationSession, opponentModel, params);
		setDecoupledComponents(acceptConditions, offeringStrategy, opponentModel, omStrategy);
		
		// Get reservation value and max bid to determine relevant range of bins
		double minUtility;
		double maxUtility;
		
		try {
			Bid maxUtilBid = this.utilitySpace.getMaxUtilityBid();
			Bid minUtilBid = this.utilitySpace.getMinUtilityBid();
			maxUtility = this.utilitySpace.getUtility(maxUtilBid);
			minUtility = this.utilitySpace.getUtility(minUtilBid);
			minUtility = Math.max(minUtility, this.utilitySpace.getReservationValueUndiscounted());
		} catch (Exception e) {
			// exception is thrown by getMaxUtilityBid if there are no bids in the outcomespace
			// but I guess that's pretty rare. Default to 0.0 - 1.0 to prevent crashes.
			maxUtility = 1.0;
			minUtility = 0.0;
		}

		int minBin = this.getBinIndex(minUtility);
		int maxBin = this.getBinIndex(maxUtility);

		offeringStrategy.setMinMaxBin(new Range(minBin, maxBin));
	}

	@Override
	public String getName() {
		return "Q-learner";
	}

	/**
	 * @param negoEvent
	 * @return AbstractState object representing the current state of the agent
	 * given the negoEvent
	 */
	public AbstractState getStateRepresentation(MultipartyNegoActionEvent negoEvent) {
		Bid oppLastBid = negotiationSession.getOpponentBidHistory().getLastBid();
		Bid myLastBid = negotiationSession.getOwnBidHistory().getLastBid();
		Bid agreement = negoEvent.getAgreement();
		Action currentAction = negoEvent.getAction();

		if (agreement != null || currentAction.getClass() == EndNegotiation.class || negoEvent.getTime() == 1.0) {
			return State.TERMINAL;
		}

		int myBin = this.getBinIndex(myLastBid);
		int oppBin = this.getBinIndex(oppLastBid);

		double time = negotiationSession.getTime();

		State state = new State(myBin, oppBin, this.getTimeBinIndex(time));

		return state;
	}

	/**
	 * Checks if a valid bid is passed and calculates the bin in which it would fall
	 * otherwise return extreme number to indicate that the bid doesn't exist yet.
	 * 
	 * @param bid
	 * @return
	 */
	protected int getBinIndex(Bid bid) {
		int bin;
		if (bid != null) {
			double bidUtil = this.getUtility(bid);
			bin = this.getBinIndex(bidUtil);
		} else {
			bin = Integer.MIN_VALUE;
		}

		return bin;
	}

	/**
	 * Helper function that calculates the bin index based
	 * on a specified utility. Is called by the similarly
	 * named function that takes a bid as argument.
	 *
	 * @param util
	 * @return
	 */
	private int getBinIndex(double util) {
		util = Math.min(0.999, util); // ensures maximum bid is in bin (N_BINS - 1)
		int n_bins = offeringStrategy.getNBins();
		return (int) Math.floor(util * n_bins);
	}

	/**
	 *
	 * @param time
	 * @return bin index that represents the current time in the state. Is binned
	 * based on a strategy parameter
	 */
	protected int getTimeBinIndex(double time) {
		return (int) Math.floor(time * this.getStrategyParameters().getValueAsDouble("time_bins"));
	}

	@Override
	public double getReward(Bid agreement) {
		double reward = 0.0;
		
		if (agreement != null) {
			reward = this.getUtility(agreement);
		}

		return reward;
	}

	@Override
	public void observeEnvironment(double reward, AbstractState newState) {
		this.offeringStrategy.observeEnvironment(reward, newState);

		if (newState.isTerminalState()) {
			this.writeObjectToFile(this.offeringStrategy.getQTable());
		}
	}

	public void writeObjectToFile(Object serObj) {
		
		String pathToQtable = this.strategyParameters.getValueAsString("_path_to_qtable");
		String filepath = parsePathToQtable(pathToQtable);

		File outputFile = new File(filepath);
		outputFile.getParentFile().mkdirs();
		try {
			FileOutputStream fileOut = new FileOutputStream(filepath);
			ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
			objectOut.writeObject(serObj);
			objectOut.close();
			System.out.println("The Q-table was succesfully written to a file");

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public Object readObjectFromFile(String filepath) {
		Object obj = null;

		try {
			FileInputStream fileIn = new FileInputStream(filepath);
			ObjectInputStream objectIn = new ObjectInputStream(fileIn);
			obj = objectIn.readObject();
			objectIn.close();
			System.out.println("Succesfully read object");
		} catch (Exception ex) {
			if (ex instanceof FileNotFoundException) {
				System.out.println("qTable file does not exist. A new file will be created.");
			} else {
				ex.printStackTrace();
			}
		}

		return obj;
	}
	
	private String parsePathToQtable(String rawPath) {
		String filepath;
		File check = new File(rawPath);
		if (check.isFile() || rawPath.endsWith(".table")) {
			filepath = rawPath;
		} else {
			filepath = rawPath + this.instanceIdentifier();
		}
		return filepath;
	}

	public String instanceIdentifier() {
		String domainName = this.negotiationSession.getDomain().getName().replace(".xml", "").replace("etc/templates", "").replace("/", "_");
		return String.format("%s-%s-%s-%s", this.getName(), domainName,
				this.utilitySpace.getFileName().replace('/', '_').replace(domainName, ""),
				this.filterStrategyParameters(this.getStrategyParameters()).replace(';', '-').replace('=', '@')); // because ; is csv																						// delimiter);
	}

	public String filterStrategyParameters(StrategyParameters parameters) {
		String fullString = parameters.toString();
		String filteredString = "";

		// filter out pairs that look like this: _key=value
		// these are not part of the identifier. Convention adapted from
		// pythonic 'private variable' indication (self._privatevar)
		for (String pair : fullString.split(";")) {
			if (!pair.startsWith("_") && pair.contains("=")) {
				filteredString = filteredString + ";" + pair;
			}
		}

		return filteredString;
	}
}
