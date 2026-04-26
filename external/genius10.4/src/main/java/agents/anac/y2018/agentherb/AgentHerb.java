package agents.anac.y2018.agentherb;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

public class AgentHerb extends AbstractNegotiationParty{
    private final Map<AgentID, BidHistory> agentsBidHistories = new HashMap<>();
    private Bid lastOfferedBid = null;
    private BidHistory initialHistory = new BidHistory();
    private final VectorConverter vectorConverter = new VectorConverter();

    private class BidHistory extends ArrayList<Tuple<Bid, Boolean>> {}

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);
        if (this.getData().getPersistentDataType() == PersistentDataType.STANDARD) {
            StandardInfoList infoList = (StandardInfoList) info.getPersistentData().get();
            for (StandardInfo sessionInfo : infoList) {
                Bid initialBid = sessionInfo.getAgreement().get1();
                if (initialBid != null) {
                    System.out.println(String.format("initial bid: %s", initialBid.toString()));
                    initialHistory.add(new Tuple<>(initialBid, true));
                }
            }
        }
    }

    /**
     * Receive message and save it in the bid history
     * If it is accept save the bid the agent accept with true
     * if it is offer save the last bid with false since the agent did not accept and
     * save the bid he made in his offer with true assuming because he offered it he would accept it
     *
     * @param sender The id of the agent who sent the message
     * @param act the action that was sent
     */
    @Override
    public void receiveMessage(AgentID sender, Action act) {
        super.receiveMessage(sender, act);
        if (act instanceof Offer || act instanceof Accept) {
            if (!agentsBidHistories.containsKey(sender)) {
                agentsBidHistories.put(sender, (BidHistory) initialHistory.clone());
            }
        }

        if (act instanceof Offer) {
            Bid bid = ((Offer) act).getBid();
            this.agentsBidHistories.get(sender).add(new Tuple<>(bid, true));
            if (this.lastOfferedBid != null) {
                this.agentsBidHistories.get(sender).add(new Tuple<>(this.lastOfferedBid, false));
            }
            this.lastOfferedBid = bid;
        } else if (act instanceof Accept) {
            Bid bid = ((Accept) act).getBid();
            this.agentsBidHistories.get(sender).add(new Tuple<>(bid, true));
        }
    }

    /**
     * Choose if to accept the last offer or make a new offer
     *
     * First we initialize a logistic regression model for each agent (except us) with his bid history
     * and whether he accepted each bid or rejected it
     * We train a new logistic regression each time and not use the same one for the whole session
     * because retrain it each time gives a better results
     * (probably because of the random weight before you start train it)
     *
     * Then for each bid we value the chances each agent will accept it using the logistic regression models
     * and evaluate the bid by multiple them all together with the utility of the bid and choose the bid
     * with the highest evaluation
     *
     * Then if the last offered bid has an higher utility we accept it, otherwise we offer the bid we chose
     *
     * @param list The available actions to do
     * @return The chosen action
     */
    @Override
    public Action chooseAction(List<Class<? extends Action>> list) {
        try {
            System.out.println(getPartyId());
            System.out.println("choosing action");

            System.out.println("initializing model");
            List<LogisticRegression> logisticRegressionsModels = this.initializeModels();

            System.out.println("searching for best bid");
            Bid nextBid = this.findNextBid(logisticRegressionsModels);

            System.out.println("choosing of accepting or offering");

            if (list.contains(Accept.class) && shouldAccept(nextBid))  {
                System.out.println("Accepting");
                return new Accept(getPartyId(), lastOfferedBid);
            } else {
                System.out.println("offering");
                this.lastOfferedBid = nextBid;
                return new Offer(this.getPartyId(), nextBid);
            }

        } catch (Exception e) {
            e.printStackTrace();
            return new EndNegotiation(getPartyId());
        }
    }

    /**
     * @param nextBid The next bid to offer
     * @return Whether to accept the last bid or offer the nextBid
     */
    private boolean shouldAccept(Bid nextBid) {
        if (lastOfferedBid != null) {
            System.out.println(String.format("last bid utility %f", this.utilitySpace.getUtility(this.lastOfferedBid)));
            if (this.getUtility(lastOfferedBid) >= this.getUtility(nextBid) * this.utilitySpace.getDiscountFactor()) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param logisticRegressionsModels The models of the agents
     * @return The next bid to offer
     */
    private Bid findNextBid(List<LogisticRegression> logisticRegressionsModels) {
        double bestBidEvaluation = 0;
        Bid nextBid = null;

        BidIterator bidIterator = new BidIterator(this.utilitySpace.getDomain());

        while (bidIterator.hasNext()) {
            Bid bid = bidIterator.next();
            Vector vector = this.vectorConverter.convert(bid);
            double chancesForAcceptance = 1;
            for (LogisticRegression model : logisticRegressionsModels) {
                chancesForAcceptance *= model.classify(vector);
            }
            double bidUtility = this.utilitySpace.getUtility(bid);
            double bidEvaluation = bidUtility + chancesForAcceptance;

            if (bidEvaluation >= bestBidEvaluation) {
                nextBid = bid;
                bestBidEvaluation = bidEvaluation;
            }
        }
        System.out.println(String.format("next bid evaluation %f", bestBidEvaluation));
        System.out.println(String.format("next bid utility %f", this.utilitySpace.getUtility(nextBid)));
        return nextBid;
    }

    /**
     * Initialize the models of the agents
     * each model gets a bid and returns the chances the agent will accept it
     *
     * @return The logistic models of the agents
     */
    private List<LogisticRegression> initializeModels() {
        List<LogisticRegression> logisticRegressionsModels = new ArrayList<>();
        for (BidHistory bidHistory : this.agentsBidHistories.values()) {
            LogisticRegression logisticRegression = new LogisticRegression(
                    this.vectorConverter.getVectorSize(this.utilitySpace.getDomain()));
            for (Tuple<Bid, Boolean> bidToDidAccept : bidHistory) {
                Vector vector = this.vectorConverter.convert(bidToDidAccept.get1());
                int label = bidToDidAccept.get2() ? 1 : 0;
                logisticRegression.train(vector, label);
            }
            logisticRegressionsModels.add(logisticRegression);
        }
        return logisticRegressionsModels;
    }

    @Override
    public String getDescription() {
        return "ANAC2018";
    }
}
