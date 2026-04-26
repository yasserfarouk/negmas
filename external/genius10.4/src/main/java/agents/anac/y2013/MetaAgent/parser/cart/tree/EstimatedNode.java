package agents.anac.y2013.MetaAgent.parser.cart.tree;

import java.util.Vector;

public class EstimatedNode extends Node{

	private double _estimated_rate; 
	protected EstimatedNode(int _id, int _observations, int _leftId,
			int _rightId, int _leftobs, int _rightObs, double _complexityParam,
			double estimated_rate, double mean_deviance, Vector<PrimarySplit> _primary_splits,
			Vector<SurrogateSplit> _Surrogate_splits) {
		super(_id, _observations, _leftId, _rightId, _leftobs, _rightObs,
				_complexityParam,_primary_splits, _Surrogate_splits);
		_estimated_rate=estimated_rate;
	}
	public static EstimatedNode factory(String text){
		String [] texts=text.trim().split(" ");
		
		int id=Integer.parseInt(texts[0].substring(0,texts[0].indexOf(":")));
		int observations=Integer.parseInt(texts[1]);;
		double complex=-1;
		if(text.contains("complexity"))
			complex=Double.parseDouble(substring(text,"complexity param="," "));	
		double estimated=Double.parseDouble(substring(text,"estimated rate=",","));
		String m=substring(text,"mean deviance=","\n");
		double meanDeviance=Double.parseDouble(m);
		int leftid=-1;
		if(text.contains("left son"))
			leftid=Integer.parseInt(substring(text,"left son="," "));
		
		
		int rightId=-1;
		if(text.contains("right son"))
			rightId=Integer.parseInt(substring(text,"right son="," "));
			
		int leftObs=-1;
		if(text.contains("left"))
			leftObs=parseObs(text,"left");
		int rightObs=-1;
		if(text.contains("right"))
			rightObs=parseObs(text,"right");
		
		if(text.contains("Surrogate")){
			String []prims=substring(text,"Primary splits:\n","Surrogate splits:\n").trim().split("\n");
			
			Vector<PrimarySplit> primary_splits=parsePrimarySplits(prims);
			String []Surrogate=substring(text,"Surrogate splits:\n",text.length()).trim().split("\n");
			Vector<SurrogateSplit> Surrogate_splits=parseSurrogateSplits(Surrogate);
			EstimatedNode node=new EstimatedNode(id, observations, leftid, rightId, leftObs, rightObs, complex, estimated, meanDeviance, primary_splits,Surrogate_splits);
			return node;
		}
		else if(text.contains("Primary")){
			String []prims=substring(text,"Primary splits:\n",text.length()).trim().split("\n");
			Vector<PrimarySplit> primary_splits=parsePrimarySplits(prims);
			EstimatedNode node=new EstimatedNode(id, observations, leftid, rightId, leftObs, rightObs, complex, estimated, meanDeviance, primary_splits,null);
			return node;
		}
		return new EstimatedNode(id, observations, leftid, rightId, leftObs, rightObs, complex, estimated, meanDeviance, null,null);
	}

	@Override
	public double get_mean() {
		return -1;
	}
	@Override
	public double get_estimated_rate() {
		return this._estimated_rate;
	}
}
