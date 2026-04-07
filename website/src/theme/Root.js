import React from 'react';

import NeuralNetworkBackground from '../components/NeuralNetworkBackground';

export default function Root({children}) {
  return (
    <>
      <NeuralNetworkBackground />
      <div className="site-content-layer">{children}</div>
    </>
  );
}