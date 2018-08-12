from sample_players import DataPlayer
from isolation.isolation import _WIDTH, _HEIGHT
import random


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 4:
            self.queue.put(random.choice(state.actions()))
        else:
            for i in range(1, 6):
                m = self.minimax_alpha_beta(state, i)
                if m is not None:
                    self.queue.put(m)

    def minimax_alpha_beta(self, state, depth):
        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move =  state.actions()[0]
        for a in state.actions():
            v = min_value(state.result(a), depth - 1, alpha, beta)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
    
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        ########################################################################
        # The below was one of my heuristcis where i tried to figure out the 
        # distance to the edges
        #x_own, y_own = own_loc%13, own_loc//13
        #d = min(x_own, _WIDTH - 1 - x_own, y_own, _HEIGHT - 1 - y_own)
        
        #if (d >= 2):
        #    multiplier = 1
        #else:
        #    multiplier = 2
        
        ########################################################################
        
        # Tried several multiplier values and chnaged this from 1 - 3 to gauge results.
        multiplier = 2
        
        ########################################################################
        # The below is a heuristic where i change the agressiveness from 3 to 2 to 1
        
        #if state.ply_count <= 2:
        #    multiplier = 3
        #elif state.ply_count > 2 and state.ply_count < 8:
        #    multiplier = 2
        ########################################################################
        
        diff = len(own_liberties) - multiplier*len(opp_liberties)

        return diff