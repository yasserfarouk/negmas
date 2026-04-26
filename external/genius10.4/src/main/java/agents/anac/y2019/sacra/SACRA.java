package agents.anac.y2019.sacra;

import agents.anac.y2019.sacra.yacomponents.BidUtilComparator;
import agents.anac.y2019.sacra.yacomponents.BidUtility;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

import java.util.*;

public class SACRA extends AbstractNegotiationParty {
    private final boolean DEBUG = false;
    private final static int DEFAULT_NUMBER_OF_CANDIDATES = 20000;
    private final double TEMPERATURE_ALPHA = 0.5;
    private final int SA_NUMBER_OF_ITERATION = 10000;
    private final double NEIGHBOR_WEIGHT_RANGE = 0.1;

    /**
     * Determines the ratio of weight change in the process of generating a neighbor utility space
     */
    private final double NEIGHBOR_CHANGE_RATIO = 0.5;

    protected int numberOfCandidates = -1;
    protected List<BidUtility> candidateOffers;
    private Bid firstReceivedBid = null;
    private Bid lastReceivedBid = null;

    public void init(NegotiationInfo info) {
        Set<BidUtility> candidateSet;
        super.init(info);

        if (numberOfCandidates < 0)
            numberOfCandidates = DEFAULT_NUMBER_OF_CANDIDATES;

        this.candidateOffers = new ArrayList<>();
        for (int i = 0; i < numberOfCandidates; i++)
            candidateOffers.add(generateRandomBidUtility());

        candidateSet = new HashSet<>(candidateOffers);
        candidateOffers = new ArrayList<>(candidateSet);
        candidateOffers.sort(new BidUtilComparator().reversed());
    }

    @Override
    public void receiveMessage(AgentID sender, Action action) {
        super.receiveMessage(sender, action);
        if (action instanceof Offer) {
            Bid bid = ((Offer)action).getBid();
            if (firstReceivedBid == null)
                firstReceivedBid = bid;
            lastReceivedBid = bid;
        }
    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        double concessionRate;
        double targetUtility;
        double acceptProbability;
        Bid maxUtilityBid;
        Bid offerBid;

        try {
            maxUtilityBid = utilitySpace.getMaxUtilityBid();
        }
        catch (Exception e) {
            maxUtilityBid = getNearestCandidate(1);
        }

        if (lastReceivedBid == null || firstReceivedBid == null) {
            return new Offer(getPartyId(), maxUtilityBid);
        }

        concessionRate = Math.max(0, utilitySpace.getUtility(lastReceivedBid) - utilitySpace.getUtility(firstReceivedBid))
                / utilitySpace.getUtility(maxUtilityBid) * 0.7;
        targetUtility = utilitySpace.getUtility(maxUtilityBid) - concessionRate;
        acceptProbability = (utilitySpace.getUtility(lastReceivedBid) - targetUtility)
                / (utilitySpace.getUtility(maxUtilityBid) - targetUtility);

        if (DEBUG)
            System.out.printf("EstimatedUtil: %.3f, concessionRate: %.3f, targetUtil: %.3f, acceptProb: %.3f\n",
                    utilitySpace.getUtility(lastReceivedBid),
                    concessionRate,
                    targetUtility,
                    acceptProbability);

        if (this.rand.nextDouble() < acceptProbability) {
            return new Accept(getPartyId(), lastReceivedBid);
        }

        offerBid = this.getCandidateAboveUtil(targetUtility);
return new Offer(getPartyId(), offerBid);
    }

    @Override
    public String getDescription() {
        return "Simulated Annealing-based Concession Rate controlling Agent";
    }

    @Override
    public AbstractUtilitySpace estimateUtilitySpace() {
        AdditiveUtilitySpace currentUtilitySpace = generateRandomUtilitySpace();
        double currentEnergy = getEnergy(currentUtilitySpace);

        AdditiveUtilitySpace nextUtilitySpace;
        double nextEnergy;

        AdditiveUtilitySpace bestUtilitySpace = currentUtilitySpace;
        double bestEnergy = currentEnergy;

        for (int nIteration = 0; nIteration < SA_NUMBER_OF_ITERATION; nIteration++) {
            nextUtilitySpace = generateNeighborUtilitySpace(currentUtilitySpace);
            nextEnergy = getEnergy(nextUtilitySpace);

            if (nextEnergy < bestEnergy) {
                bestUtilitySpace = nextUtilitySpace;
                bestEnergy = nextEnergy;
            }

            if (this.rand.nextDouble() > getOverrideProbability(
                    currentEnergy, nextEnergy, getTemperature(nIteration / SA_NUMBER_OF_ITERATION))) {
                currentUtilitySpace = nextUtilitySpace;
                currentEnergy = nextEnergy;
            }
        }

        return bestUtilitySpace;
    }

    /** Returns "energy" of the additive utility space (Lower is better)
     * @param additiveUtilitySpace
     * @return energy
     */
    private double getEnergy(AdditiveUtilitySpace additiveUtilitySpace) {
        return -getAdditiveUtilitySpaceScore(additiveUtilitySpace);
    }

    /**
     * Returns score of the additive utility space (Higher is better)
     * @param additiveUtilitySpace
     * @return score
     */
    private double getAdditiveUtilitySpaceScore(AdditiveUtilitySpace additiveUtilitySpace) {
        BidRanking bidRank = this.userModel.getBidRanking();
        Map<Bid, Integer> realRanks = new HashMap();
        List<Double> estimatedUtils = new ArrayList();
        Map<Bid, Integer> estimatedRanks = new HashMap();

        for (Bid bid : bidRank.getBidOrder()) {
            realRanks.put(bid, realRanks.size());
            estimatedUtils.add(additiveUtilitySpace.getUtility(bid));
        }
        Collections.sort(estimatedUtils);

        for (Bid bid : bidRank.getBidOrder()) {
            estimatedRanks.put(bid, estimatedUtils.indexOf(additiveUtilitySpace.getUtility(bid)));
        }

        double errors = 0.0D;

        for (Bid bid : bidRank.getBidOrder()) {
            errors += Math.pow((double)(realRanks.get(bid) - estimatedRanks.get(bid)), 2);
        }

        double spearman = 1.0D - 6.0D * errors / (Math.pow((double)realRanks.size(), 3.0D) - (double)realRanks.size());
        double lowDiff = Math.abs(bidRank.getLowUtility() - additiveUtilitySpace.getUtility(bidRank.getMinimalBid()));
        double highDiff = Math.abs(bidRank.getHighUtility() - additiveUtilitySpace.getUtility(bidRank.getMaximalBid()));

        return spearman * 10.0D + (1.0D - lowDiff) + (1.0D - highDiff);
    }

    private double getTemperature(double r) {
        return Math.pow(TEMPERATURE_ALPHA, r);
    }

    private double getOverrideProbability(double oldEnergy, double newEnergy, double temperature) {
        if (newEnergy <= oldEnergy)
            return 1.0;
        return Math.exp((oldEnergy - newEnergy) / temperature);
    }

    /**
     * Generate a neighbor of baseUtilitySpace (only a weight or a value of an issue is changed)
     * @param baseUtilitySpace
     * @return neighborUtilitySpace
     */
    private AdditiveUtilitySpace generateNeighborUtilitySpace(AdditiveUtilitySpace baseUtilitySpace) {
        AdditiveUtilitySpaceFactory neighborFactory = new AdditiveUtilitySpaceFactory(this.getDomain());
        List<IssueDiscrete> issueList = neighborFactory.getIssues();
        IssueDiscrete targetIssue = issueList.get(this.rand.nextInt(issueList.size()));
        boolean isChangeWeight = this.rand.nextDouble() < NEIGHBOR_CHANGE_RATIO;

        for (IssueDiscrete issue : neighborFactory.getIssues()) {
            neighborFactory.setWeight(issue, baseUtilitySpace.getWeight(issue));

            for (ValueDiscrete value : issue.getValues()) {
                neighborFactory.setUtility(issue, value,
                        ((EvaluatorDiscrete)baseUtilitySpace.getEvaluator(issue)).getDoubleValue(value));
            }
        }

        if (isChangeWeight) {
            neighborFactory.setWeight(targetIssue,
                    Math.max(0, baseUtilitySpace.getWeight(targetIssue)
                                + (this.rand.nextDouble() - 0.5) * NEIGHBOR_WEIGHT_RANGE / issueList.size()));
        } else {
            ValueDiscrete targetValue = targetIssue.getValue(this.rand.nextInt(targetIssue.getNumberOfValues()));
            double evaluatedValue = ((EvaluatorDiscrete)baseUtilitySpace.getEvaluator(targetIssue)).getDoubleValue(targetValue);

            neighborFactory.setUtility(targetIssue, targetValue,
                    Math.max(0, evaluatedValue * (1 + (this.rand.nextDouble() - 0.5) * NEIGHBOR_WEIGHT_RANGE)));
        }

        neighborFactory.normalizeWeights();
        return neighborFactory.getUtilitySpace();
    }

    private AdditiveUtilitySpace generateRandomUtilitySpace() {
        AdditiveUtilitySpaceFactory randomFactory = new AdditiveUtilitySpaceFactory(this.getDomain());

        for (IssueDiscrete issue : randomFactory.getIssues()){
            randomFactory.setWeight(issue, this.rand.nextDouble());
            for (ValueDiscrete value : issue.getValues()) {
                randomFactory.setUtility(issue, value, this.rand.nextDouble());
            }
        }

        randomFactory.normalizeWeights();
        return randomFactory.getUtilitySpace();
    }

    protected BidUtility generateRandomBidUtility() {
        return new BidUtility(generateRandomBid(), utilitySpace);
    }

    protected Bid getNearestCandidate(double util) {
        int index = getNearestCandidateIndex(util);
        return candidateOffers.get(index).getBid();
    }

    protected Bid getCandidateAboveUtil(double util) {
        int maxIndex = getNearestCandidateIndex(util);
        int index = this.rand.nextInt(maxIndex);
        return candidateOffers.get(index).getBid();
    }

    protected int getNearestCandidateIndex(double util) {
        int index;
        int rangeMin = 0;
        int rangeMax = candidateOffers.size() - 1;

        do {
            index = rangeMin + (rangeMax - rangeMin) / 2;
            if (util < candidateOffers.get(index).getUtil())
                rangeMin = index;
            else
                rangeMax = index;
        } while (rangeMax - rangeMin > 1);

        return index;
    }

}
