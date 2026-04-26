package agents.anac.y2018.ponpokorampage;

import java.util.List;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.UtilitySpace;


public class PonPokoRampage extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private Domain domain = null;

	private Map<String, List<Bid>> hoge;
	private Map<String, Boolean>  hardliner;

	private List<BidInfo> lBids;

	private double threshold_low = 0.99;
	private double threshold_high = 1.0;

	private final int PATTERN_SIZE = 5;
	private int pattern = 0;
//	PrintStream out;
	private int count = 0;

	@Override
	public void init(NegotiationInfo info) {

//		try {
//			out = new PrintStream("C:\\debug.log");
//		}catch (Exception e){
//
//		}

		super.init(info);
		this.domain = info.getUtilitySpace().getDomain();

//		long a = System.currentTimeMillis();
		lBids = new ArrayList<>(AgentTool.generateRandomBids(this.domain, 30000, this.rand, this.utilitySpace ) );
		Collections.sort(lBids, new BidInfoComp().reversed() );
//		out.println(System.currentTimeMillis() -a);
		hoge = new HashMap<>();
		hardliner = new HashMap<>();


		pattern = rand.nextInt(PATTERN_SIZE);
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		count++;
		long a = System.currentTimeMillis();

		if (utilitySpace.getReservationValue() >= 0.9){
			return new EndNegotiation(getPartyId());
		}

		if (count % 200 == 199){
			hoge.forEach( (k ,v) -> {
				//out.println((new HashSet<>(v)).size());
				if( (new HashSet<>(v)).size() < 20 * getTimeLine().getTime()){
					hardliner.put(k,Boolean.TRUE);
				}
				else {
					hardliner.put(k, Boolean.FALSE);
				}
			});
		}

		//譲歩度合いの設定
		if (pattern == 0) {
			threshold_high = 1 - 0.1 * timeline.getTime();
			threshold_low = 1 - 0.2 * timeline.getTime() - 0.1 * Math.abs(Math.sin(this.timeline.getTime() * 16));
		}
		else if (pattern == 1){
			threshold_high = 1;
			threshold_low = 1 - 0.22 * timeline.getTime();
		}
		else if (pattern == 2){
			threshold_high = 1 - 0.1 * timeline.getTime();
			threshold_low = 1 - 0.1 * timeline.getTime() - 0.15 * Math.abs(Math.sin(this.timeline.getTime() * 32));
		}
		else if (pattern == 3){
			threshold_high = 1 - 0.05 * timeline.getTime();
			threshold_low = 1 - 0.2 * timeline.getTime();
			if (timeline.getTime() > 0.98){
				threshold_low = 1 - 0.3 * timeline.getTime();
			}
		}else if (pattern == 4){
			threshold_high = 1 - 0.15 * this.timeline.getTime() * Math.abs(Math.sin(this.timeline.getTime() * 32));
			threshold_low = 1 - 0.25 * this.timeline.getTime() * Math.abs(Math.sin(this.timeline.getTime() * 32));
		}
		else {
			System.out.println("pattern error");
			threshold_high = 1 - 0.1 * timeline.getTime();
			threshold_low = 1 - 0.2 * Math.abs(Math.sin(this.timeline.getTime() * 64));
		}

		if (hardliner.containsValue(Boolean.TRUE)){
			threshold_low += 0.05;
		}

		//Accept判定
		if ( lastReceivedBid != null) {
			if (getUtility(lastReceivedBid) > threshold_low ){
				return new Accept(getPartyId(), lastReceivedBid);
			}
		}

		//Offerするbidの選択
		Bid bid = null;
		while (bid == null){
			bid = AgentTool.selectBidfromList(this.lBids, this.threshold_high, this.threshold_low);
			if (bid == null) {
				threshold_low -= 0.001;
			}
		}
		return new Offer(getPartyId(), bid);
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
			if (hoge.containsKey(sender.getName())){
				hoge.get(sender.getName()).add(((Offer) action).getBid());
			}else {
				hoge.put(sender.getName(), new ArrayList<>());
				hoge.get(sender.getName()).add( ((Offer) action).getBid());
			}
		}
	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

}

class AgentTool{

	private static Random random = new Random();

	public static Bid selectBidfromList( List<BidInfo> bidInfoList, double higerutil, double lowwerutil){
		List<BidInfo> bidInfos = new ArrayList<>();
		bidInfoList.forEach(bidInfo -> {
			if (bidInfo.getutil() <= higerutil && bidInfo.getutil() >= lowwerutil){
				bidInfos.add(bidInfo);
			}
		});
		if (bidInfos.size() == 0){
			return null;
		}
		else {
			return bidInfos.get(random.nextInt(bidInfos.size())).getBid();
		}
	}

	public static Set<BidInfo> generateRandomBids(Domain d, int numberOfBids, Random random, UtilitySpace utilitySpace){
		Set<BidInfo> randombids = new HashSet<>();
		for (int i = 0; i < numberOfBids; i++){
			Bid b = d.getRandomBid(random);
			randombids.add(new BidInfo(b, utilitySpace.getUtility(b)));
		}
		return randombids;
	}

}


//bidとその効用を保存するクラス
class BidInfo{
	Bid bid;
	double util;
	public BidInfo( Bid b ){
		this.bid = b;
		util = 0.0;
	}
	public BidInfo( Bid b , double u){
		this.bid = b;
		util = u;
	}

	public void setutil(double u){
		util = u;
	}

	public Bid getBid(){
		return bid;
	}

	public double getutil(){
		return util;
	}


	@Override
	public int hashCode() {
		return bid.hashCode();
	}

	public boolean equals(BidInfo bidInfo) {
		return bid.equals(bidInfo.getBid());
	}

	@Override
	public boolean equals(Object obj){
		if ( obj == null){
			return false;
		}
		if ( obj instanceof BidInfo){
			return ((BidInfo)obj).getBid().equals(bid);
		}else {
			return false;
		}
	}

}

final class BidInfoComp implements Comparator<BidInfo> {
	BidInfoComp(){
		super();
	}
	@Override
	public int compare(BidInfo o1, BidInfo o2) {
		return Double.compare(o1.getutil(), o2.getutil());
	}
}
