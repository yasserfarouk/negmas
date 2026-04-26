package agents.anac.y2018.libra;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;

import negotiator.parties.BoulwareNegotiationParty;
import negotiator.parties.ConcederNegotiationParty;

import agents.anac.y2015.Atlas3.Atlas3;
//
import agents.anac.y2015.ParsAgent.ParsAgent;
import agents.anac.y2015.RandomDance.RandomDance;
import agents.anac.y2016.caduceus.Caduceus;
import agents.anac.y2016.farma.Farma;
import agents.anac.y2016.myagent.MyAgent;
import agents.anac.y2016.parscat.ParsCat;
//
import agents.anac.y2016.terra.Terra;
//
import agents.anac.y2016.yxagent.YXAgent;
//
import agents.anac.y2017.agentf.AgentF;
import agents.anac.y2017.agentkn.AgentKN;
import agents.anac.y2017.caduceusdc16.CaduceusDC16;
import agents.anac.y2017.mamenchis.Mamenchis;
//
import agents.anac.y2017.parsagent3.ShahAgent;
import agents.anac.y2017.parscat2.ParsCat2;
import agents.anac.y2017.ponpokoagent.PonPokoAgent;
import agents.anac.y2017.rubick.Rubick;
//
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
//
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
//
/**
 * This is your negotiation party.
 */
public class Libra extends AbstractNegotiationParty {

	HashMap<AgentID,Bid> lastReceivedBidMap = new HashMap<AgentID, Bid>();
	private Bid lastReceivedBid = null;
	private Bid lastOfferedBid = null;
	public  int agentNum = 19;
	public NegotiationParty[] agents = new NegotiationParty[agentNum];
	public double[] weightS = new double[agentNum];
	private double def_weight =10.0;
	private double weightSum = (double)agentNum*def_weight;
	private double chg_weight = 2.0;
	private double min_weight = 1.0;
	private Action[] lastActionS = new Action[agentNum];
	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		System.out.println("Discount Factor is " + info.getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is " + info.getUtilitySpace().getReservationValueUndiscounted());

		// if you need to initialize some variables, please initialize them
		// below
		agents[0] = new BoulwareNegotiationParty();
		agents[1] = new ConcederNegotiationParty();
		//
		agents[2] = new ParsAgent();
		agents[3] = new RandomDance();
		agents[4] = new Atlas3();
		//
		agents[5] = new YXAgent();
		agents[6] = new Farma();
		agents[7] = new ParsCat();
		//
		agents[8] = new ShahAgent();
		agents[9] = new PonPokoAgent();
		agents[10] = new Mamenchis();
		//
		agents[11] = new Rubick();
		agents[12] = new AgentKN();
		agents[13] = new ParsCat2();
		//
		agents[14] = new Terra();
		agents[15] = new Caduceus();
		agents[16] = new MyAgent();
		//
		agents[17] = new AgentF();
		agents[18] = new CaduceusDC16();
		//
		int index = 0;
		for(NegotiationParty agent : agents){
			agent.init(info);
			weightS[index] = def_weight;
			index+=1;
		}
		//weightS[16] = 20;
		System.out.println("Init has finished");
	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		System.out.println("ChooseAction has started");
		double offerVote = 0;
		double acceptVote = 0;
		double endVote = 0;
		int index = 0;
		ArrayList<Bid> offeredList = new ArrayList<Bid>();
		ArrayList<Double> weightList = new ArrayList<Double>();
		for(NegotiationParty agent : agents){
			Action act = agent.chooseAction(validActions);
			lastActionS[index] = act;
			if (act instanceof Offer){
				offerVote+=weightS[index];
				Bid b =((Offer) act).getBid();
				//Offer proposed = new Offer(getPartyId(),b);
				offeredList.add(b);
				weightList.add(weightS[index]);
			}
			else if (act instanceof Accept){
				acceptVote+=weightS[index];
			}
			else {
				endVote+=weightS[index];
			}
			index+=1;
		}
		System.out.println("offerVote="+offerVote);
		System.out.println("acceptVote="+acceptVote);
		System.out.println("endVote="+endVote);
		//
		if(offerVote>acceptVote && offerVote>endVote){
			//System.out.println("Offer is chosen");
			Bid newBid = generateRandomBid();
			System.out.println("Before="+newBid);
			HashMap<Integer,Value>valueMap = newBid.getValues();
			double receivedAve = 0;
			if(lastReceivedBidMap.isEmpty()){
				receivedAve = 0.5;
			}
			else {
				for (AgentID sender : lastReceivedBidMap.keySet()) {
					receivedAve += getUtilityWithDiscount(lastReceivedBidMap.get(sender));
				}
				receivedAve/=lastReceivedBidMap.size();
			}

			for(Integer issueKey : valueMap.keySet()){
				//HashMap<Value,Integer> valueCount = new HashMap<Value, Integer>();
				HashMap<Value,Double> valueCount = new HashMap<Value, Double>();
				index = 0;
				//System.out.println("Index is reset");
				//System.out.println(offeredList.size());
				for(Bid proposed : offeredList){
					Value proposedValue = proposed.getValue(issueKey);
					//System.out.println("Value is picked");
					if(valueCount.containsKey(proposedValue)){
						//Integer count = valueCount.get(issueKey);
						//valueCount.put(proposedValue,count+1);
						//System.out.println("Value is editing");
						double current = valueCount.get(proposedValue);
						valueCount.put(proposedValue,current+weightList.get(index)*(getUtilityWithDiscount(proposed)-receivedAve)*100);
						//System.out.println("Value is edited");
					}
					else{
						//valueCount.put(proposedValue,1);
						valueCount.put(proposedValue,weightList.get(index)*(getUtilityWithDiscount(proposed)-receivedAve)*100);
						//System.out.println("Value is set");
					}
					//System.out.println("Value is picked2");
					index+=1;
				}
				//System.out.println("Count has finished");
				double maxCount = 0;
				Value bestValue = null;
				for(Value valueKey : valueCount.keySet()){
					//int count = valueCount.get(valueKey);
					double count = valueCount.get(valueKey);
					if(count>maxCount){
						bestValue = valueKey;
						maxCount = count;
					}
				}
				newBid = newBid.putValue(issueKey,bestValue);
				//System.out.println("Choice has finished");
			}
			System.out.println("Offer is Decided");
			lastOfferedBid = newBid;
			System.out.println("After="+newBid);
			return  new Offer(getPartyId(),newBid);
		}
		else if(acceptVote>offerVote && acceptVote>endVote){
			System.out.println("Accept is Decided");
			return new Accept(getPartyId(), lastReceivedBid);
		}
		else{
			System.out.println("EndNegotiation is Decided");
			return  new EndNegotiation(getPartyId());
		}
		/*
		// with 50% chance, counter offer
		// if we are the first party, also offer.
		if (lastReceivedBid == null || !validActions.contains(Accept.class) || Math.random() > 0.5) {
			return new Offer(getPartyId(), generateRandomBid());
		} else {
			return new Accept(getPartyId(), lastReceivedBid);
		}
		//*/
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action. Can be null.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
			lastReceivedBidMap.put(sender,((Offer) action).getBid());
		}
		for(NegotiationParty agent : agents){
			agent.receiveMessage(sender, action);
		}
		if(lastOfferedBid==null){
			System.out.println("ReceiveMessage has finished with return");
			return;
		}
		//
		System.out.println("ReceiveMessage has started");
		if (action instanceof Offer){//相手が別の提案をしてきた場合
			//System.out.println("Received Action is Offer");
			int index = 0;
			Bid b =((Offer) action).getBid();
			double receive_util = getUtilityWithDiscount(b);
			double offered_util = getUtilityWithDiscount(lastOfferedBid);
			System.out.println(lastActionS.length);
			for(Action lastAct:lastActionS){
				if(lastAct instanceof  Offer){
					//System.out.println("Last Action is offer");
					if(receive_util>offered_util*1.1){//自分以上の提案をもらえた場合
						weightS[index]-=chg_weight;
					}
					else{//自分以下の場合
						weightS[index]+=chg_weight;
					}
				}
				else if(lastAct instanceof  Accept){
					//System.out.println("Last Action is Accept");
					if(receive_util>offered_util*1.1){//自分以上の提案をもらえた場合
						weightS[index]+=chg_weight;
					}
					else{//自分以下の場合
						weightS[index]-=chg_weight;
						if(weightS[index]<min_weight){
							weightS[index] = min_weight;
						}
					}
				}
				else{
					//System.out.println("Last Action is EndNegotiation");
					//double reserve_util = getUtilitySpace().getReservationValue();
					double reserve_util = getUtilitySpace().getReservationValueWithDiscount(getTimeLine());
					if(receive_util>=reserve_util){//留保価格以上の提案をもらえた場合
						weightS[index]-=chg_weight;
						if(weightS[index]<min_weight){
							weightS[index] = min_weight;
						}
					}
					else{//留保価格以下の場合
						weightS[index]+=chg_weight;
					}
				}
				index+=1;
			}
		}
		else if(action instanceof Accept){//相手が受け入れた場合
			//System.out.println("Received Action is Accept");
			int index = 0;
			for(Action lastAct:lastActionS){
				if(lastAct instanceof Offer){
					weightS[index]+=chg_weight;
				}
				else{
					weightS[index]-=chg_weight;
					if(weightS[index]<min_weight){
						weightS[index] = min_weight;
					}
				}
				index+=1;
			}
		}
		//
		//System.out.println("Normalization has started");
		normalizeWeightS();
		System.out.println("ReceiveMessage has finished");
		return;
	}

	public void normalizeWeightS(){
		//System.out.println("Summation has started");
		double weightSum2 = 0;
		for(int index3=0;index3<weightS.length;index3++){
			//System.out.println(weightS[index3]);
			weightSum2+=weightS[index3];
			//System.out.println(weightSum2);
			//System.out.println(index3);
		}
		//System.out.println("Summation has finished");
		for(int index3=0;index3<weightS.length;index3++){
			weightS[index3] = weightS[index3]*(weightSum/weightSum2);
		}
		//System.out.println("Normalization has started");
		return;
	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

}
