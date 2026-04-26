package agents.anac.y2012.MetaAgent.agents.WinnerAgent;

import java.util.HashMap;
import java.util.Vector;

public class Loop<T> {
	
	// this class is helpful for creating all possible offers in the domain
	private Loop<T> _nextLoop;
	private Vector<T> _vec;
	private HashMap<Integer, Integer> _indexMap;

	public Loop(Vector<T> vec, HashMap<Integer, Integer> indexMap){
		_vec = vec;
		_indexMap = indexMap;
		_nextLoop = null;
	}
	
	public Loop(Vector<T> vec, HashMap<Integer, Integer> indexMap, Loop<T> next){
		this(vec, indexMap);
		_nextLoop = next;
	}
	
	public Loop<T> setNext(Vector<T> vec) {
		_nextLoop = new Loop<T>(vec, _indexMap);
		return _nextLoop;
	}
	
	public void iteration(HashMap<Integer,T> outerLoop, int index,Vector<HashMap<Integer,T>> cartesianProd, int limit){
		for(T o : _vec) {
			if(cartesianProd.size() >= limit) {
				break;
			}
			HashMap<Integer,T> innerMap = new HashMap<Integer,T>();
			innerMap.putAll(outerLoop);
			innerMap.put(_indexMap.get(index), o);
			if(_nextLoop != null) {
				_nextLoop.iteration(innerMap, index+1, cartesianProd, limit);
			}
			else {
				cartesianProd.add(innerMap);
			}
		}
	}
}

