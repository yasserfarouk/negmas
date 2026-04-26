package genius.core.misc;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * Class which generates the Cartesian product of a list of sets.
 * Can be optimized by using an iterative approach.
 * 
 * @author Mark Hendrikx
 */
public class SetTools {

	/**
	 * Given a list of sets, this method returns the Cartesian product
	 * of the given sets.
	 * @param <A> class of object contained in set.
	 * @param sets sets of objects which Cartesian product must be determined.
	 * @return set of sets symbolizing the Cartesian product
	 */
	public static <A> Set<Set<A>> cartesianProduct(Set<A>... sets) {
	    if (sets.length < 2) {
	    	Iterator<A> setIterator = sets[0].iterator();
	    	Set<Set<A>> mainSet = new HashSet<Set<A>>();
	    	while (setIterator.hasNext()) {
	    		A item = setIterator.next();
	    		Set<A> set = new HashSet<A>();
	    		set.add(item);
	    		mainSet.add(set);
	    	}
	    	return mainSet;
	    }
	    return _cartesianProduct(0, sets);
	}

	private static <A> Set<Set<A>> _cartesianProduct(int index, Set<A>... sets) {
	    Set<Set<A>> ret = new HashSet<Set<A>>();
	    if (index == sets.length) {
	        ret.add(new HashSet<A>());
	    } else {
	        for (A obj : sets[index]) {
	            for (Set<A> set : _cartesianProduct(index+1, sets)) {
	                set.add(obj);
	                ret.add(set);
	            }
	        }
	    }
	    return ret;
	}
}