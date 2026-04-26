package agents.anac.y2018.exp_rubick;


import java.util.List;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;






public class Exp_Rubick extends AbstractNegotiationParty{
	
	public Exp_Rubick()
    {
        parties = new LinkedList();
		lastReceivedBid = null;
		frequentValuesList0 = new LinkedList();
	    frequentValuesList1 = new LinkedList();
        profileOrder = new LinkedList();
        isHistoryAnalyzed = false;
        numberOfReceivedOffer = 0;
        histOpp1 = new LinkedList();
        histOpp0 = new LinkedList();
        lastpartyname = "";
        sos = null;
        maxReceivedBidutil = 0.0D;
        opponentNames = new String[2];
        bestAcceptedBids = new LinkedList();
        threshold = 0.0D;
        opp0bag = new ArrayList();
        opp1bag = new ArrayList();
    }
	
	public void init(NegotiationInfo info)
    {
        super.init(info);
        double time = info.getTimeline().getTime();
        List array = new ArrayList();
        for(int i = 0; i <= 1000; i++)
        {
            if(i < 1)
                Bid = 0.10000000000000001D;
            else
                array.add(Double.valueOf(Bid));
            if(i > 0)
                Prob = ((Double)array.get(i - 1)).doubleValue();
            System.out.println(Bid);
        }
        
            double sum = 0.0D;
            
            for(int i = 0; i < array.size(); i++)
            {
                sum += ((Double)array.get(i)).doubleValue();
                Average = sum / (double)array.size();
                if(i > 0)
                    Agent = Average;
            }

            subborn = Bid - Prob;
            subborn1 = Math.abs(Bid - Prob);
            emax = Agent + (1.0D - Agent) * subborn1;
            target = 1.0D - (1.0D - emax) * Math.pow(time, 3D);
            concession = target - Bid;
            target1 = 1.0D - (1.0D - emax) * Math.pow(time, (2.7000000000000002D + concession) - time);
            System.out.println(target1);
            info.getUtilitySpace().getDomain().getIssues().size();
        
        String domainName = utilitySpace.getFileName();
        domainName = domainName.substring(domainName.lastIndexOf("/") + 1, domainName.lastIndexOf("."));
        sos = new SortedOutcomeSpace(utilitySpace);
        sortPartyProfiles(getPartyId().toString());
        history = (StandardInfoList)getData().get();
        threshold = info.getUtilitySpace().getReservationValue().doubleValue();
        maxReceivedBidutil = threshold;
        initializeOpponentModelling();
        
        
        
    }
	
	public void initializeOpponentModelling()
    {
        int issueSize = utilitySpace.getDomain().getIssues().size();
        for(int i = 0; i < issueSize; i++)
        {
            LinkedHashMap valueAmountMap = new LinkedHashMap();
            frequentValuesList0.add(valueAmountMap);
            valueAmountMap = new LinkedHashMap();
            frequentValuesList1.add(valueAmountMap);
            int issuecnt = 0;
            for(Iterator iterator = utilitySpace.getDomain().getIssues().iterator(); iterator.hasNext(); issuecnt++)
            {
                Issue issue = (Issue)iterator.next();
                IssueDiscrete issued = (IssueDiscrete)issue;
                if(issuecnt != i)
                    continue;
                Value value;
                for(Iterator iterator1 = issued.getValues().iterator(); iterator1.hasNext(); ((LinkedHashMap)frequentValuesList1.get(i)).put(value, Integer.valueOf(0)))
                {
                    value = (Value)iterator1.next();
                    ((LinkedHashMap)frequentValuesList0.get(i)).put(value, Integer.valueOf(0));
                }

            }

        }

    }
	
	public Action chooseAction(List validActions)
    {

        double time = timeline.getTime();
        List array = new ArrayList();
        for(int i = 0; i <= 1000; i++)
        {
            if(i < 1)
                Bid = 0.10000000000000001D;
            else
                array.add(Double.valueOf(Bid));
            if(i > 0)
                Prob = ((Double)array.get(i - 1)).doubleValue();
            System.out.println(Bid);
        }

        double sum = 0.0D;
        for(int i = 0; i < array.size(); i++)
        {
            sum += ((Double)array.get(i)).doubleValue();
           Average = sum / (double)array.size();
            if(i > 0)
                Agent = Average;
        }

        subborn = Bid - Prob;
        subborn1 = Math.abs(Bid - Prob);
        emax = Agent + (1.0D - Agent) * subborn;
        target = 1.0D - (1.0D - emax) * Math.pow(time, 3D);
        concession = target - Bid;
        target1 = 1.0D - (1.0D - emax) * Math.pow(0.80000000000000004D * time, (3D + 2D * concession) - 0.80000000000000004D * time);
        System.out.println(target1);
        if(getUtility(lastReceivedBid) >= target1)
            return new Accept(getPartyId(), lastReceivedBid);
        Bid newBid = generateBid(target1);
        int i = 999;
        do
        {
            if(i <= 0 && lastReceivedBid == null && lastReceivedBid != null)
                break;
            newBid = generateBid(target1);
            if(getUtility(newBid) >= target1)
                break;
            i--;
        } while(true);
        return new Offer(getPartyId(), newBid);
    }

	
	public void receiveMessage(AgentID sender, Action action)
    {
        super.receiveMessage(sender, action);
        if(action instanceof Accept)
        {
            Bid acceptedBid = ((Accept)action).getBid();
            if(bestAcceptedBids.isEmpty())
                bestAcceptedBids.add(acceptedBid);
            else
            if(!bestAcceptedBids.contains(acceptedBid))
            {
                int size = bestAcceptedBids.size();
                for(int i = 0; i < size; i++)
                {
                    if(getUtility(acceptedBid) > getUtility((Bid)bestAcceptedBids.get(i)))
                    {
                        bestAcceptedBids.add(i, acceptedBid);
                        break;
                    }
                    if(i == bestAcceptedBids.size() - 1)
                        bestAcceptedBids.add(acceptedBid);
                }

            }
        }
        if(action instanceof Offer)
        {
            lastReceivedBid = ((Offer)action).getBid();
            if(maxReceivedBidutil < getUtilityWithDiscount(lastReceivedBid))
                maxReceivedBidutil = getUtilityWithDiscount(lastReceivedBid) * 0.94999999999999996D;
            numberOfReceivedOffer++;
            String partyName = getPartyName(action.getAgent().toString());
            lastpartyname = partyName;
            BidResolver(lastReceivedBid, partyName);
            if(!parties.contains(partyName))
                sortPartyProfiles(action.getAgent().toString());
            if(parties.size() == 3 && !history.isEmpty() && !isHistoryAnalyzed)
                analyzeHistory();
        }
    }
	
	private int takeTheChance(double maxReceived)
    {
        int pow = 1;
        double chance = rand.nextDouble();
        if(chance > 0.94999999999999996D + 0.050000000000000003D * maxReceived)
            pow = 2;
        else
        if(chance > 0.93000000000000005D + 0.070000000000000007D * maxReceived)
            pow = 3;
        else
            pow = 10;
        return pow;
    }
	

	
	private Bid generateBid(double targetutil)
    {
        Bid bid = null;
        if(timeline.getTime() > 0.995D && !bestAcceptedBids.isEmpty())
        {
            int s = bestAcceptedBids.size();
            if(s > 3)
                s = 3;
            int ind = rand.nextInt(s);
            bid = (Bid)bestAcceptedBids.get(ind);
        } else
        {
            if(opp0bag.size() > 0 && opp1bag.size() > 0)
                bid = searchCandidateBids(targetutil);
            if(bid == null)
                bid = sos.getBidNearUtility(targetutil).getBid();
        }
        System.out.flush();
        return bid;
    }
	
	
	public Bid searchCandidateBids(double targetutil)
    {
        double bu = 0.0D;
        Value valu = null;
        LinkedList intersection = new LinkedList();
        LinkedList candidateBids = new LinkedList();
        Iterator iterator = sos.getAllOutcomes().iterator();
        do
        {
            if(!iterator.hasNext())
                break;
            BidDetails bd = (BidDetails)iterator.next();
            bu = getUtility(bd.getBid());
            if(bu < targetutil)
                break;
            int score = 0;
            for(int isn = 0; isn < bd.getBid().getIssues().size(); isn++)
            {
                valu = bd.getBid().getValue(isn + 1);
                if(valu == opp0bag.get(isn))
                    score++;
                if(valu == opp1bag.get(isn))
                    score++;
            }

            intersection.add(Integer.valueOf(score));
            candidateBids.add(bd.getBid());
        } while(true);
        int max = -1;
        for(int i = 0; i < intersection.size(); i++)
            if(max < ((Integer)intersection.get(i)).intValue())
                max = i;

        if(candidateBids.size() > 1)
            return (Bid)candidateBids.get(max);
        else
            return null;
    }
	
	public void BidResolver(Bid bid, String partyname)
    {
        Value valu = null;
        if(partyname.equals(opponentNames[0]))
        {
            for(int isn = 0; isn < bid.getIssues().size(); isn++)
            {
                valu = bid.getValue(isn + 1);
                int prevAmount = ((Integer)((LinkedHashMap)frequentValuesList0.get(isn)).get(valu)).intValue();
                ((LinkedHashMap)frequentValuesList0.get(isn)).put(valu, Integer.valueOf(prevAmount + 1));
            }

        } else
        if(partyname.equals(opponentNames[1]))
        {
            for(int isn = 0; isn < bid.getIssues().size(); isn++)
            {
                valu = bid.getValue(isn + 1);
                int prevAmount = ((Integer)((LinkedHashMap)frequentValuesList1.get(isn)).get(valu)).intValue();
                ((LinkedHashMap)frequentValuesList1.get(isn)).put(valu, Integer.valueOf(prevAmount + 1));
            }

        }
        if(numberOfReceivedOffer > 2)
            extractOpponentPreferences();
    }
	
	
	public void extractOpponentPreferences()
    {
        ArrayList opp0priors = new ArrayList();
        ArrayList opp1priors = new ArrayList();
        opp0bag = new ArrayList();
        opp1bag = new ArrayList();
        LinkedList meanEvalValues0 = new LinkedList();
        LinkedList meanEvalValues1 = new LinkedList();
        for(int i = 0; i < frequentValuesList0.size(); i++)
        {
            double sum = 0.0D;
            for(Iterator iterator1 = ((LinkedHashMap)frequentValuesList0.get(i)).keySet().iterator(); iterator1.hasNext();)
            {
                Value val = (Value)iterator1.next();
                sum += ((Integer)((LinkedHashMap)frequentValuesList0.get(i)).get(val)).intValue();
            }

            meanEvalValues0.add(Double.valueOf(sum / (double)frequentValuesList0.size()));
            sum = 0.0D;
            for(Iterator iterator2 = ((LinkedHashMap)frequentValuesList1.get(i)).keySet().iterator(); iterator2.hasNext();)
            {
                Value val = (Value)iterator2.next();
                sum += ((Integer)((LinkedHashMap)frequentValuesList1.get(i)).get(val)).intValue();
            }

            meanEvalValues1.add(Double.valueOf(sum / (double)frequentValuesList1.size()));
        }

        for(int i = 0; i < frequentValuesList0.size(); i++)
        {
            Iterator iterator = ((LinkedHashMap)frequentValuesList0.get(i)).keySet().iterator();
            do
            {
                if(!iterator.hasNext())
                    break;
                Value val = (Value)iterator.next();
                if((double)((Integer)((LinkedHashMap)frequentValuesList0.get(i)).get(val)).intValue() >= ((Double)meanEvalValues0.get(i)).doubleValue())
                    opp0priors.add(val);
            } while(true);
            opp0bag.add(opp0priors.get(rand.nextInt(opp0priors.size())));
            opp0priors = new ArrayList();
            iterator = ((LinkedHashMap)frequentValuesList1.get(i)).keySet().iterator();
            do
            {
                if(!iterator.hasNext())
                    break;
                Value val = (Value)iterator.next();
                if((double)((Integer)((LinkedHashMap)frequentValuesList1.get(i)).get(val)).intValue() >= ((Double)meanEvalValues1.get(i)).doubleValue())
                    opp1priors.add(val);
            } while(true);
            opp1bag.add(opp1priors.get(rand.nextInt(opp1priors.size())));
            opp1priors = new ArrayList();
        }

    }
	
	
	private Integer extractPartyID(String partyID)
    {
        return Integer.valueOf(Integer.parseInt(partyID.substring(partyID.indexOf("@") + 1, partyID.length())));
    }
	
	private void analyzeHistory()
    {
        isHistoryAnalyzed = true;
        for(int h = 0; h <= history.size() - 1; h++)
        {
            LinkedList utilsOp1 = new LinkedList();
            LinkedList utilsOp2 = new LinkedList();
            StandardInfo info = (StandardInfo)history.get(h);
            boolean historyMatch = true;
            int cnt = 0;
            for(Iterator iterator = info.getUtilities().iterator(); iterator.hasNext(); cnt++)
            {
                Tuple offered = (Tuple)iterator.next();
                String partyname = getPartyName((String)offered.get1());
                Double util = (Double)offered.get2();
                if(cnt < 3 && !partyname.equals(parties.get(cnt)))
                {
                    historyMatch = false;
                    break;
                }
                if(partyname.equals(opponentNames[0]))
                {
                    utilsOp1.add(util);
                    if(util.doubleValue() > acceptanceLimits[0])
                        acceptanceLimits[0] = util.doubleValue();
                    continue;
                }
                if(!partyname.equals(opponentNames[1]))
                    continue;
                utilsOp2.add(util);
                if(util.doubleValue() > acceptanceLimits[1])
                    acceptanceLimits[1] = util.doubleValue();
            }

        }

    }
	
	private void sortPartyProfiles(String partyID)
    {
        int pid = extractPartyID(partyID).intValue();
        String partyName = getPartyName(partyID);
        if(profileOrder.isEmpty())
        {
            profileOrder.add(Integer.valueOf(pid));
            parties.add(partyName);
        } else
        {
            int size = profileOrder.size();
            for(int id = 0; id < size; id++)
            {
                if(pid < ((Integer)profileOrder.get(id)).intValue())
                {
                    profileOrder.add(id, Integer.valueOf(pid));
                    parties.add(id, partyName);
                    break;
                }
                if(id == profileOrder.size() - 1)
                {
                    profileOrder.add(Integer.valueOf(pid));
                    parties.add(partyName);
                }
            }

        }
        int p = 0;
        Iterator iterator = parties.iterator();
        do
        {
            if(!iterator.hasNext())
                break;
            String party = (String)iterator.next();
            if(!party.equals(getPartyName(getPartyId().toString())))
            {
                opponentNames[p] = party;
                p++;
            }
        } while(true);
    }
	
	private String getPartyName(String partyID)
    {
        return partyID.substring(0, partyID.indexOf("@"));
    }
	
    @Override
    public String getDescription() {
        return "ANAC2018";
    }
	
	 private Bid lastReceivedBid;
	    private StandardInfoList history;
	    private LinkedList parties;
	    private LinkedList histOpp0;
	    private LinkedList histOpp1;
	    private boolean isHistoryAnalyzed; 
	    private double Bid;
	    private double Prob;
	    private double Average;
	    private double Agent;
	    private double concession;
	    private double subborn;
	    private double subborn1;
	    private double target;
	    private double target1;
	    private double emax;
	    private int numberOfReceivedOffer;
	    private LinkedList profileOrder;
	    private String opponentNames[];
	    private double acceptanceLimits[] = {
	        0.0D, 0.0D
	    };
	    private double maxReceivedBidutil;
	    private String lastpartyname;
	    private SortedOutcomeSpace sos;
	    private LinkedList bestAcceptedBids;
	    private double threshold;
	    protected LinkedList frequentValuesList0;
	    protected LinkedList frequentValuesList1;
	    ArrayList opp0bag;
	    ArrayList opp1bag;
}
