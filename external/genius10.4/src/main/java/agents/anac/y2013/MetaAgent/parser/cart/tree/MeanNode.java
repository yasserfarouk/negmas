package agents.anac.y2013.MetaAgent.parser.cart.tree;

import java.util.Vector;

public class MeanNode extends Node {
		public double _mean;
		public double _MSE;
		private MeanNode(int id, int observations, int leftId, int rightId,
				int leftobs, int rightObs, double complexityParam, double mean,
				double MSE, Vector<PrimarySplit> primary_splits,
				Vector<SurrogateSplit> Surrogate_splits) {
			super(id, observations, leftId, rightId, leftobs, rightObs, complexityParam,primary_splits,Surrogate_splits);
			this._mean=mean;
			this._MSE=MSE;
		}		
		
		public static MeanNode factory(String text){
			String [] texts=text.trim().split(" ");
			
			int id=Integer.parseInt(texts[0].substring(0,texts[0].indexOf(":")));
			int observations=Integer.parseInt(texts[1]);;
			double complex=-1;
			if(text.contains("complexity"))
				complex=Double.parseDouble(substring(text,"complexity param="," "));	
			double mean=Double.parseDouble(substring(text,"mean=",","));
			String m=substring(text,"MSE=","\n");
			double mse=Double.parseDouble(m);
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
				MeanNode node=new MeanNode(id, observations, leftid, rightId, leftObs, rightObs, complex, mean, mse, primary_splits,Surrogate_splits);
				return node;
			}
			else if(text.contains("Primary")){
				String []prims=substring(text,"Primary splits:\n",text.length()).trim().split("\n");
				Vector<PrimarySplit> primary_splits=parsePrimarySplits(prims);
				MeanNode node=new MeanNode(id, observations, leftid, rightId, leftObs, rightObs, complex, mean, mse, primary_splits,null);
				return node;
			}
			return new MeanNode(id, observations, leftid, rightId, leftObs, rightObs, complex, mean, mse, null,null);
		}

		@Override
		public double get_mean() {
			return this._mean;
		}

		@Override
		public double get_estimated_rate() {
			// TODO Auto-generated method stub
			return -1;
		}
		
		
	}
