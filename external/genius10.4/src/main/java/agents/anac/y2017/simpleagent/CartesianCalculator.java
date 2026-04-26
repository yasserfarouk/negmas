package agents.anac.y2017.simpleagent;

import java.util.ArrayList;
import java.util.List;

public class CartesianCalculator {

	public <T> List<List<T>> calculate(List<List<T>> input) {
		List<List<T>> res = new ArrayList<>();
		if (input.isEmpty()) { // if no more elements to process
			res.add(new ArrayList<T>()); // then add empty list and return
			return res;
		} else {
			process(input, res); // we need to calculate the cartesian product
									// of input and store it in res variable
		}
		return res; // method completes , return result
	}

	private <T> void process(List<List<T>> lists, List<List<T>> res) {
		List<T> head = lists.get(0); // take first element of the list
		List<List<T>> tail = calculate(lists.subList(1, lists.size())); // invoke
																		// calculate
																		// on
																		// remaining
																		// element,
																		// here
																		// is
																		// recursion

		for (T h : head) { // for each head
			for (List<T> t : tail) { // iterate over the tail
				List<T> tmp = new ArrayList<>(t.size());
				tmp.add(h); // add the head
				tmp.addAll(t); // and current tail element

				res.add(tmp);
			}
		}
	}
}