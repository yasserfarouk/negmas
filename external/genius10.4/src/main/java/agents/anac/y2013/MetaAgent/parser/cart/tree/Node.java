package agents.anac.y2013.MetaAgent.parser.cart.tree;
import java.util.HashMap;
import java.util.Vector;


public abstract class Node {

int _id;
int _observations;
int _leftId;
int _rightId;
int _leftobs;
int _rightObs;
Node _left;
Node _right;
double _complexityParam;
Vector<PrimarySplit> _primary_splits;
Vector<SurrogateSplit> _Surrogate_splits;

enum treeType{
	MEAN,ESTIMATED
}

protected Node(int _id, int _observations, int _leftId, int _rightId,
		int _leftobs, int _rightObs, double _complexityParam,Vector<PrimarySplit> _primary_splits,
		Vector<SurrogateSplit> _Surrogate_splits) {
	super();
	this._id = _id;
	this._observations = _observations;
	this._leftId = _leftId;
	this._rightId = _rightId;
	this._leftobs = _leftobs;
	this._rightObs = _rightObs;
	this._complexityParam = _complexityParam;
	this._primary_splits = _primary_splits;
	this._Surrogate_splits = _Surrogate_splits;
}


protected static Vector< PrimarySplit> parsePrimarySplits(String[] splitsText) {
	Vector< PrimarySplit> splits=new Vector<PrimarySplit>();
	for (String string : splitsText) {
		PrimarySplit s=PrimarySplit.factory(string);
		splits.add(s );
	}
	return splits;
}

protected static Vector< SurrogateSplit> parseSurrogateSplits(String[] splitsText) {
	Vector< SurrogateSplit> splits=new Vector<SurrogateSplit>();
	for (String string : splitsText) {
		SurrogateSplit s=SurrogateSplit.factory(string);
		splits.add(s );
	}
	return splits;
}


public int get_id() {
	return _id;
}
public int get_leftId() {
	return _leftId;
}
public int get_rightId() {
	return _rightId;
}
public Node get_left() {
	return _left;
}
public void set_left(Node _left) {
	this._left = _left;
}
public Node get_right() {
	return _right;
}
public void set_right(Node _right) {
	this._right = _right;
}
protected static int parseObs(String text, String dir) {
	text=substring(text, dir,"obs)");
	return Integer.parseInt(text.split(" ")[2].substring(1));
}
public static String substring(String s,String from,String to){
	s=s.substring(s.indexOf(from)+from.length());
	if(s.indexOf(to)!=-1)
		s=s.substring(0,s.indexOf(to));
	
	return s;
}
public static String substring(String s,String from,int to){
	return s.substring(s.indexOf(from)+from.length(),to);
}

public Node getBestNode(HashMap<String,Double> values){
	if(isLeaf()){
		return this;
	}
	else return navigate(values);
}
protected Node navigate(HashMap<String, Double> values) {
	if(this._primary_splits!=null){
		PrimarySplit.Direction dir=this._primary_splits.get(0).getDirection(values);
		switch (dir) {
		case LEFT:{
			return this._left.getBestNode(values);
		}
		case RIGHT:{
			return this._right.getBestNode(values);
		}
		default:
			break;
		} 
	}
	if(this._Surrogate_splits!=null){
		PrimarySplit.Direction dir=this._Surrogate_splits.get(0).getDirection(values);
		switch (dir) {
		case LEFT:{
//			System.out.println(this.get_id() +" choose left" );
			return this._left.getBestNode(values);
		}
		case RIGHT:{
//			System.out.println(this.get_id() +" choose right"); 
			return this._right.getBestNode(values);
		}
		default:
			break;
		} 
	}
	return null;
}
protected boolean isLeaf(){
	if(this._left==null&&this._right==null)
		return true;
	return false;
}


public abstract double get_mean();
public abstract double get_estimated_rate();
}