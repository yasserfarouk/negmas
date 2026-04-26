package agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded;


import java.util.ArrayList;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.analysis.BidPoint;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * @author W.Pasman 15nov07
 * BidSpace is a class that can store and do analysis of a space of bids.
 * There seems lot of overlap with the Analysis class.
 * But to be safe I did not try to modify the Analysis class and introduced a new class.
 */
public class BidSpace {
	AdditiveUtilitySpace utilspaceA;
	AdditiveUtilitySpace utilspaceB;
	Domain domain; // equals to utilspaceA.domain = utilspaceB.domain
	public ArrayList<BidPoint> bidPoints;
	ArrayList<BidPoint> paretoFrontier=null; // not yet set.
	BidPoint kalaiSmorodinsky=null; // null if not set.
	BidPoint nash=null; // null if not set.
	
	public BidSpace(AdditiveUtilitySpace spaceA, AdditiveUtilitySpace spaceB) throws Exception
	{
		utilspaceA=spaceA;
		utilspaceB=spaceB;
		if (utilspaceA==null || utilspaceB==null)
			throw new NullPointerException("util space is null");
		domain=utilspaceA.getDomain();
		utilspaceA.checkReadyForNegotiation(domain);
		utilspaceB.checkReadyForNegotiation(domain);
		BuildSpace();
	}
	
	
	/**
	 * special version that does NOT check the *second* utility space for
	 * compatibility with the domain. Use on your own risk.
	 * The first space must contain the domain.
	 * @param spaceA
	 * @param spaceB
	 * @param anything. If you use this three-para initializer the check will not
	 * be done on 2nd domain. The boolean has no function at all except
	 * being a third parameter that makes a differnet function call.
	 * @throws Exception
	 */
	public BidSpace(AdditiveUtilitySpace spaceA, AdditiveUtilitySpace spaceB,boolean anything) throws Exception
	{
		utilspaceA=spaceA;
		utilspaceB=spaceB;
		if (utilspaceA==null || utilspaceB==null)
			throw new NullPointerException("util space is null");
		domain=utilspaceA.getDomain();
		utilspaceA.checkReadyForNegotiation(domain);
		BuildSpace();
	}
	
	
	/*
	 * Create the space with all bid points from the two util spaces.
	 * @throws exception if utility can not be computed for some poitn.
	 * This should not happen as it seems we checked beforehand that all is set OK.
	 */
	void BuildSpace() throws Exception
	{
		bidPoints=new ArrayList<BidPoint>();
		BidIterator lBidIter = new BidIterator(domain);
		
		while(lBidIter.hasNext()) {
			Bid bid = lBidIter.next();
			bidPoints.add(new BidPoint(null, utilspaceA.getUtility(bid), utilspaceB.getUtility(bid)));
		}

	}
	
	/**
	 * 
	 * @return pareto frontier. The order is  ascending utilityA.
	 * @throws Exception
	 */
	public ArrayList<BidPoint> getParetoFrontier() throws Exception
	{
		ArrayList<BidPoint> subPareto = new ArrayList<BidPoint >();
		boolean bIsBidSpaceAvailable = true;
		if(bidPoints.size()<1) bIsBidSpaceAvailable =false; 
		if (paretoFrontier==null)
		{		
			BidIterator lBidIter = new BidIterator(domain);
			int count=0;
			ArrayList<BidPoint> tmpBidPoints;
			if(bIsBidSpaceAvailable)
				tmpBidPoints = new ArrayList<BidPoint>(bidPoints);
			else
				tmpBidPoints = new ArrayList<BidPoint>();
			while(lBidIter.hasNext()) {
				Bid bid = lBidIter.next();
//				System.out.println(bid.toString());				
				if(!bIsBidSpaceAvailable) tmpBidPoints.add(new BidPoint(bid,utilspaceA.getUtility(bid),utilspaceB.getUtility(bid)));
				count++;
				if(count>500000) {
					subPareto.addAll(computeParetoFrontier(tmpBidPoints));
					tmpBidPoints = new ArrayList<BidPoint >();
					count = 0;
				}
			}
			if(tmpBidPoints.size()>0)subPareto.addAll(computeParetoFrontier(tmpBidPoints));
		       //System.out.println("ParetoFrontier start computation:"+(new Date()));
	        paretoFrontier=computeParetoFrontier(subPareto);
	        //System.out.println("ParetoFrontier end computation:"+(new Date()));

		}

 		
		return paretoFrontier;
	}
	
	/** private because it should be called only
	 * with the bids as built by BuildSpace.
	 * @param points the ArrayList<BidPoint> as computed by BuildSpace and stored in bidpoints.
	 * @throws Exception if problem occurs
	 * @return the pareto frontier of the bidpoints.
	 */
	ArrayList<BidPoint> computeParetoFrontier(ArrayList<BidPoint> points) throws Exception
	{
		int n=points.size();
		if (n<=1) return points; // end recursion
		
		// split list in two halves. Unfortunately ArrayList does not have support for this...
		// make new lists that can be modified by us.
		ArrayList<BidPoint> points1=new ArrayList<BidPoint>();
		ArrayList<BidPoint> points2=new ArrayList<BidPoint>();
		for (int i=0; i<n/2; i++) points1.add(points.get(i));
		for (int i=n/2; i<n; i++) points2.add(points.get(i));
		
		ArrayList<BidPoint> pareto1=computeParetoFrontier(points1);
		ArrayList<BidPoint> pareto2=computeParetoFrontier(points2);
		return mergeParetoFrontiers(pareto1,pareto2);
	}
	
	/**
	 * @author W.Pasman
	 * @param pareto1 the first pareto frontier: list of bidpoints with increasing utility for A, decreasing for B
	 * @param pareto2 the second pareto frontier:...
	 * @return new pareto frontier being the merged frontier of the two. Sorted in increasing utilityA direction
	 */
	public ArrayList<BidPoint> mergeParetoFrontiers(ArrayList<BidPoint> pareto1,ArrayList<BidPoint> pareto2)
	{
		if (pareto1.size()==0) return pareto2;
		if (pareto2.size()==0) return pareto1;

		 // clone because we will remove elements from the list but we want to keep the orig lists.
		 // This looks bit ugly....
		ArrayList<BidPoint> list1=(ArrayList<BidPoint>)(pareto1.clone()); 
		ArrayList<BidPoint> list2=(ArrayList<BidPoint>)(pareto2.clone());
		 // make sure that the first pareto list has the left most point.
		if (list1.get(0).getUtilityA()>list2.get(0).getUtilityA()) 
		{
			ArrayList<BidPoint> list3;
			list3=list1; list1=list2; list2=list3; // swap list1,list2......
		}
		
		// sort the rest
		BidPoint firstpoint=list1.remove(0);
		ArrayList<BidPoint> newpareto=mergeParetoFrontiers(list1,list2);
		 // determine if the first point of list1 can be kept.
		 // the only criterium is the first point of list 2, 
		 // it must be OK with list 1 because that is already a pareto frontier.
		if (firstpoint.getUtilityB()>list2.get(0).getUtilityB()) { 
				 // left point must be higher than next
				newpareto.add(0,firstpoint);
		}
		
		return newpareto;
	}
	
	public ArrayList<Bid> getParetoFrontierBids() throws Exception
	{
		ArrayList<Bid> bids=new ArrayList<Bid> ();
		ArrayList<BidPoint> points=getParetoFrontier();
		for (BidPoint p:points) bids.add(p.getBid());
		return bids;
	}
	
	
	/**
	 * Calculates Kalai-Smorodinsky optimal outcome. Assumes that Pareto frontier is already built.
	 * Kalai-Smorodinsky is the point on paretofrontier 
	 * that has least difference in utilities for A and B
	 * @author Dmytro Tykhonov, cleanup by W.Pasman
	 * @returns the kalaiSmorodinsky BidPoint.
	 * @throws AnalysisException
	 */
	public BidPoint getKalaiSmorodinsky() throws Exception 
	{	
		if (kalaiSmorodinsky!=null) return kalaiSmorodinsky;
		if(getParetoFrontier().size()<1) 
			throw new Exception("kalaiSmorodinsky product: Pareto frontier is unavailable.");
		double minassymetry=2; // every point in space will have lower assymetry than this.
		for (BidPoint p:paretoFrontier)
		{
			double asymofp = Math.abs(p.getUtilityA()-p.getUtilityB());
			if (asymofp<minassymetry) { kalaiSmorodinsky=p; minassymetry=asymofp; }
		}
		return kalaiSmorodinsky;
	}
	
	
	
	/**
	 * Calculates the undiscounted Nash optimal outcome. Assumes that Pareto frontier is already built.
	 * Nash is the point on paretofrontier that has max product of utilities for A and B
	 * @author Dmytro Tykhonov, cleanup by W.Pasman
	 * @returns the Nash BidPoint.
	 * @throws AnalysisException
	 */
	public BidPoint getNash() throws Exception 
	{
		if (nash!=null) return nash;
		if(getParetoFrontier().size()<1) 
			throw new Exception("Nash product: Pareto frontier is unavailable.");
		double maxp = -1;
		double agentAresValue=0, agentBresValue=0;
		if(utilspaceA.getReservationValue()!=null) agentAresValue = utilspaceA.getReservationValue();
		if(utilspaceB.getReservationValue()!=null) agentBresValue = utilspaceB.getReservationValue();
		for (BidPoint p:paretoFrontier)
		{
			double utilofp = (p.getUtilityA() -agentAresValue)*(p.getUtilityB()-agentBresValue);
			if (utilofp>maxp) { nash=p; maxp=utilofp; }
		}
		return nash;
	}
	
	/**
	 * Calculate own coordinate 
	 * @param opponentUtility
	 * @return the utility of us on the pareto curve
	 * @throws exception if getPareto fails or other cases, e.g. paretoFrontier contains utilityB=NAN.
	 * Still unclear why utilB evaluates to NAN though...
	 */
	public double OurUtilityOnPareto(double opponentUtility) throws Exception
	{
		if (opponentUtility<0. || opponentUtility>1.)
			throw new Exception("opponentUtil "+opponentUtility+" is out of [0,1].");
		ArrayList<BidPoint> pareto=getParetoFrontier();
		// our utility is along A axis, opp util along B axis.

		//add endpoints to pareto curve such that utilB spans [0,1] entirely
		if (pareto.get(0).getUtilityB()<1) pareto.add(0,new BidPoint(null,0.,1.));
		if (pareto.get(pareto.size()-1).getUtilityB()>0) pareto.add(new BidPoint(null,1.,0.));
		if (pareto.size()<2) throw new Exception("Pareto has only 1 point?!"+pareto);
		// pareto is monotonically descending in utilB direction.
		int i=0;
//		System.out.println("Searching for opponentUtility = " + opponentUtility);
		while (! (pareto.get(i).getUtilityB()>=opponentUtility && opponentUtility>=pareto.get(i+1).getUtilityB())) 
		{
//			System.out.println(i + ". Trying [" + pareto.get(i).utilityB +  ", " + pareto.get(i+1).utilityB + "]");
			i++;
		}
		
		double oppUtil1=pareto.get(i).getUtilityB(); // this is the high value
		double oppUtil2=pareto.get(i+1).getUtilityB(); // the low value
		double f=(opponentUtility-oppUtil1)/(oppUtil2-oppUtil1); // f in [0,1] is relative distance from point i.
		// close to point i means f~0. close to i+1 means f~1
		double lininterpol=(1-f)*pareto.get(i).getUtilityA()+f*pareto.get(i+1).getUtilityA();
		return lininterpol;
	}
	
	public String toString()
	{
		return bidPoints.toString();
	}
		
	/**
	 * find the bid with the minimal distance weightA*DeltaUtilA^2+weightB*DeltaUtilB^2
	 * where DeltaUtilA is the difference between given utilA and the actual util of bid
	 * @author W.Pasman
	 * @param utilA the agent-A utility of the point to be found
	 * @param utilB the agent-B utility of the point to be found
	 * @param weightA weight in A direction
	 * @param weightB weight in B direction
	 * @param excludeList Bids to be excluded from the search.
	 * @return best point, or null if none remaining.
	 */
	public BidPoint NearestBidPoint(double utilA,double utilB,double weightA,double weightB,
			ArrayList<Bid> excludeList)
	{
		System.out.println("determining nearest bid to "+utilA+","+utilB);
//		System.out.println("excludes="+excludeList);
		double mindist=9.; // paretospace distances are always smaller than 2
		BidPoint bestPoint=null;
		double r;
		for (BidPoint p:bidPoints)
		{
			boolean contains=false;
			//disabled excluding 16-11-2010
			//for (Bid b:excludeList) { if (b.equals(p.bid)) { contains=true; break; } }
			// WERKT NIET????if (excludeList.indexOf(p.bid)!=-1) continue; 
			//neither ArrayList.contains nor ArrayList.indexOf seem to use .equals
			// although manual claims that indexOf is using equals???
			if (contains) continue;
			r=weightA*sq(p.getUtilityA()-utilA)+weightB*sq(p.getUtilityB()-utilB);
			if (r<mindist) { mindist=r; bestPoint=p; }
		}
		System.out.println("point found: (" + bestPoint.getUtilityA() + ", " + bestPoint.getUtilityB() + ") ="+bestPoint.getBid());
		//System.out.println("p.bid is in excludelist:"+excludeList.indexOf(bestPoint.bid));
//		if (excludeList.size()>1) System.out.println("bid equals exclude(1):"+bestPoint.bid.equals(excludeList.get(1)));
		//System.out.println();
		return bestPoint;
	}
	
	public double sq(double x) { return x*x; }
}
